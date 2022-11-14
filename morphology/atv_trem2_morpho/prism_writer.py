#!/usr/bin/env python3

""" Tools to write Prism files

Basic Usage
-----------

The :py:class:`PrismWriter` class hides the messy details of creating a prism
file. You can add one-way tables (bar plots, box plots, etc) and two-way tables
(line plots) through a uniform interface:

.. code-block:: python

    with PrismWriter('path/to/prism/file.psfx') as writer:
        writer.add_one_way_table(
            data=df,
            group_column='Treatment',
            value_column='Values',
        )
        writer.add_two_way_table(
            data=df,
            xcolumn='Day',
            ycolumn='Treatment',
            value_column='Values',
        )

Advanced Usage
--------------

Prism files are XML documents with a long header, followed by a sequence of
XML data tables. They have to be written in several stages.

Build the header XML section of the Prism file:

.. code-block:: python

    writer = PrismBuilder()
    writer.write_created_header()
    writer.write_info_header()

One-way tables are used for categorical plots (bar charts, box plots, etc). To
order data, you need a group column (e.g. Animal ID, treatment, etc) to split the
data into categories, and a value column representing the individual measurments.

.. code-block:: python

    table_template = PrismOneWayTable(
        data=df,
        group_column='Treatment',
    )
    table_template.group_columns()
    writer.add_table('Value', table_template)

Two-way tables are used for plots where both x and y are continuous values (line plots, etc).
To order data, you need an x column for the x-axis (e.g. Timepoint, Area, etc), a group column
(e.g. Animal ID, treatment, etc) and a value column for the y-axis.

.. code-block:: python

    table_template = PrismTwoWayTable(
        data=df, xcolumn='Day',
        ycolumn='Treatment')
    table_template.group_columns()
    writer.add_table('Value', table_template)

Once you've added each table of interest, the they can be written to a file:

.. code-block:: python

    writer.write_tables()  # Convert the tables above to a single Prism XML object
    writer.save_to_file('path/to/file.pzfx')  # Actually write the XML object to disk

XML Writer class:

* :py:class:`PrismBuilder`: Write individual tables to a prism file

Main function:

* :py:class:`df_to_prism`: Convert a data frame to a prism file

Table classes:

* :py:class:`PrismTableTemplate`: Base class for Prism table templates
* :py:class:`PrismOneWayTable`: Format data as a categorical rain table for bar charts
* :py:class:`PrismTwoWayTable`: Format data as a set of category bins for line charts

Utility classes:

* :py:class:`PlaqueGrouper`: Grouper class to unpivot plaque size bins from the slide scanner
* :py:class:`ColumnEncoder`: Encode categorical values (sex, animal id, etc) as numbers

"""

# Imports
import re
import sys
import pathlib
import datetime
from collections import Counter
from typing import Optional, Dict, List

THISDIR = pathlib.Path(__file__).parent
if str(THISDIR.parent) not in sys.path:
    sys.path.insert(0, str(THISDIR.parent))

# 3rd party imports
import numpy as np

import pandas as pd

from lxml import etree

# Constants

reNUMBER = re.compile(r'[^0-9]', re.IGNORECASE)
reNORM = re.compile(r'[^0-9a-z_-]+', re.IGNORECASE)
reSPACE = re.compile(r'\s+', re.IGNORECASE)

NUM_DECIMALS = 9

HEADER = [
    "Prism group",
    "Animal ID",
    "Cohort",
    "Sex",
    "NSA Block #",
    "Section # (block of 20)",
]

# Classes


class PlaqueGrouper(object):
    """ Try to unpivot certain columns

    :param list[str] columns:
        The columns to try and regroup
    """

    def __init__(self):
        patterns = [
            r'\(?30\ ?[-_]\ ?125\ ?[µu]m\ ?2\)?',
            r'\(?125\ ?[-_]\ ?250\ ?[µu]m\ ?2\)?',
            r'\(?250\ ?[-_]\ ?500\ ?[µu]m\ ?2\)?',
            r'\(?[><]\ ?500\ ?[µu]m\ ?2\)?',
        ]
        self.patterns = [re.compile(pat, re.IGNORECASE) for pat in patterns]
        self.order = [
            '30-125 um2',
            '125-250 um2',
            '250-500 um2',
            '>500 um2',
        ]

    def pivot(self, df: pd.DataFrame,
              extra_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """ Pivot the data frame based on column name matching

        :param list[str] extra_columns:
            If not None, keep these columns too
        """
        if extra_columns is None:
            extra_columns = []
        elif isinstance(extra_columns, str):
            extra_columns = [extra_columns]
        extra_columns = [str(c) for c in extra_columns if c not in (None, '')]

        pivot_df = {
            'PlaqueGroup': [],
            'Category': [],
            'Value': [],
        }
        for col in extra_columns:
            pivot_df[col] = []
        for column in df.columns.values:
            if column in extra_columns:
                continue
            for i, pat in enumerate(self.patterns):
                match = pat.search(column)
                if not match:
                    continue
                st, ed = match.span()
                new_column = ' '.join([column[:st].strip(), column[ed:].strip()])
                new_column = new_column.strip()

                values = df[column].values
                mask = np.isfinite(values)
                values = values[mask]

                # Add all the columns and data in
                pivot_df['PlaqueGroup'].extend(self.order[i] for _ in values)
                pivot_df['Category'].extend(new_column for _ in values)
                pivot_df['Value'].extend(values)
                for col in extra_columns:
                    pivot_df[col].extend(df[col].values[mask])
        return pd.DataFrame(pivot_df)


class ColumnEncoder(object):
    """ Specialized encoders for different columns

    :param str column:
        The name of the column to encode
    """

    # Number of decimals for each representation
    NUM_DECIMALS = {
        'animal_id': 0,
        'animal': 0,
        'sex': 0,
        'prism_group': 0,
        'treatment_ip': 3,
        'nsa_block': 0,
        'section_block_of_20': 0,
    }

    # Convert string codes to numbers
    REF_CATEGORIES = {
        'sex': {
            'f': 0,
            'female': 0,
            '0': 0,
            '0_0': 0,  # Floats convert as 0.0, then '.' -> '_'
            'm': 1,
            'male': 1,
            '1': 1,
            '1_0': 1,
        },
    }

    def __init__(self, column: str):

        self.column = column
        self.column_clean = normalize_column_name(column)

        self.num_decimals = self.NUM_DECIMALS.get(self.column_clean, NUM_DECIMALS)
        self.ref_categories = self.REF_CATEGORIES.get(self.column_clean)

    def encode(self, val: object) -> str:
        """ Encode values based on the column involved

        :param object val:
            The value of the column
        :returns:
            A string encoding of that value
        """
        # Try to decode bytes-like objects
        if isinstance(val, bytes):
            val = val.decode('utf-8')

        # Look for NULL values
        if val in (np.nan, '', None):
            return ''

        # Look for typical column names
        if self.column_clean in ('animal', 'animal_id'):
            return reNUMBER.sub('', str(val))
        elif self.column_clean in ('treatment_ip', 'treatment_id'):
            # FIXME: Figure out a better way to dump the treatment ID in
            val = reNUMBER.sub('.', val).strip('.').split('.')
            if len(val) < 3:
                return '.'.join(val)
            return '.'.join([val[0], ''.join(val[1:])])

        # Use the code table to code the input value for this column
        if self.ref_categories is not None:
            return str(self.ref_categories.get(reNORM.sub('_', str(val)).lower().strip('_'), ''))

        # Generic converters
        if isinstance(val, (float, np.floating)):
            if ~np.isfinite(val):
                return ''
            return f'{val:0.16f}'
        if isinstance(val, (int, np.integer)):
            return f'{val:d}'
        return str(val).strip()


class PrismTableTemplate(object):
    """ Base class for table formats """

    @property
    def total_other_groups(self) -> List[str]:
        """ Return all possible total groups """
        groups = [None]
        for group in self.other_groups:
            if group not in groups:
                groups.append(group)
        return groups

    def get_title(self, column: str, subset: Optional[str] = None) -> str:
        """ Return the formal title for this column

        :param str column:
            The column name for this table
        :param list[str] subset:
            If not None, the list of sub-categories to use
        :returns:
            A string of the column name, and a subset marker (if any)
        """
        if self.table_title is None:
            table_title = column
        else:
            table_title = self.table_title

        if subset is None:
            return table_title
        if self.other_group_column is not None:
            clean_other = normalize_column_name(self.other_group_column)
            if clean_other == 'sex':
                if subset.startswith(('f', 'F')):
                    return f'{table_title} (females)'
                elif subset.startswith(('m', 'M')):
                    return f'{table_title} (males)'
        return f'{table_title} ({subset})'

    def group_columns(self):
        raise NotImplementedError('Implement column grouping')

    def group_data(self, column: str,
                   subset: Optional[str] = None) -> Dict[str, np.ndarray]:
        raise NotImplementedError('Implement data grouping')

    def write_table(self, column: str, table_id: str,
                    subset: Optional[str] = None) -> etree.Element:
        raise NotImplementedError('Implement table writing')


class PrismOneWayTable(PrismTableTemplate):
    """ Store the data for a one-way prism table

    :param DataFrame data:
        The data frame to extract a one-way table from
    :param str group_column:
        The column to use for y-data (e.g. prism group)
    :param str other_group_column:
        If not None, disaggregate by this column
    :param str name_column:
        The column to use for the y-column names (e.g. cohort)
    """
    def __init__(self,
                 data: pd.DataFrame,
                 group_column: str,
                 other_group_column: Optional[str] = None,
                 name_column: Optional[str] = None,
                 table_title: Optional[str] = None):
        self.group_column = group_column
        self.other_group_column = other_group_column
        self.name_column = name_column if name_column is not None else group_column

        self.table_title = table_title

        # Normalize the category values
        data = data.copy()
        for column in (group_column, other_group_column, name_column):
            if column is None:
                continue
            data.loc[:, column] = data[column].map(clean_group_values)
        self.data = data

        self.data = data
        self.groups = list(sorted(np.unique(data[group_column])))

        # FIXME: Handle coding here...
        if other_group_column is None:
            self.other_groups = [None]
        else:
            self.other_groups = list(sorted(np.unique(data[other_group_column])))
            if len(self.other_groups) < 2:
                self.other_groups = [None]

        self._other_group_index = None

        self._index = None
        self._num_replicates = None
        self._group_names = None

    def group_columns(self):
        """ Work out the mapping from column to group """
        ydata = self.data[self.group_column].values
        index = np.zeros((ydata.shape[0], 2), dtype=np.int32)
        name_data = self.data[self.name_column].values
        if self.other_group_column is None:
            other_data = None
        else:
            other_data = self.data[self.other_group_column].values

        other_group_index = {}

        # Add in a second column for alternate grouping
        group_names = {}
        total_num_replicates = 0
        for i, other_group in enumerate(self.other_groups):
            if other_group is None:
                other_mask = np.ones(ydata.shape, dtype=bool)
            else:
                other_mask = other_data == other_group

            num_replicates = -1
            for j, group in enumerate(self.groups):
                mask = np.logical_and(ydata == group, other_mask)

                if not np.any(mask):
                    continue

                names = name_data[mask]
                if not np.all(names == names[0]):
                    err = f'Got multiple levels for name "{self.name_column}" in group "{group}": {set(names)}'
                    raise ValueError(err)

                group_names[group] = names[0]

                got_reps = np.sum(mask)
                rep_index = np.arange(got_reps).astype(np.int32)

                num_replicates = max([num_replicates, got_reps])
                idx, = np.nonzero(mask)
                index[idx, 0] = rep_index + total_num_replicates
                index[idx, 1] = j

            other_group_index[other_group] = (
                total_num_replicates,
                total_num_replicates + num_replicates,
            )
            total_num_replicates += num_replicates + 1
        self._index = index
        self._group_names = group_names
        self._other_group_index = other_group_index
        self._num_replicates = total_num_replicates - 1

    def group_data(self, column: str,
                   subset: Optional[str] = None) -> Dict[str, np.ndarray]:
        """ Group data into columns

        :param str column:
            The data column to group this way
        :param str subset:
            If not None, the key from the self.other_group_column to pull out
        :returns:
            The data, stacked into blocks like a Prism OneWay table
        """
        value = self.data[column].values
        if value.dtype == object:
            block = np.full((self._num_replicates, len(self.groups)), fill_value='', dtype=object)
        else:
            block = np.full((self._num_replicates, len(self.groups)), fill_value=np.nan)

        index = self._index
        block[index[:, 0], index[:, 1]] = value
        if subset is not None:
            st, ed = self._other_group_index[subset]
            block = block[st:ed, :]

        # Force the column names to be unique
        columns = [self._group_names[g] for g in self.groups]
        column_totals = Counter(columns)
        column_counts = Counter()
        final_columns = []
        for column in columns:
            if column_totals[column] < 2:
                final_columns.append(column)
            else:
                column_counts[column] += 1
                final_columns.append(f'{column}-{column_counts[column]}')

        assert len(columns) == len(final_columns)
        assert len(final_columns) == block.shape[1]
        return {str(c): block[:, i] for i, c in enumerate(final_columns)}

    def write_table(self, column: str, table_id: str,
                    subset: Optional[str] = None) -> etree.Element:
        """ Write a table out

        :param str column:
            The column to write
        :param str table_id:
            The table ID to write as
        :param str subset:
            If not None, write only a subset of the table where the
            ``other_group_column`` matches this value (e.g. Sex == 'M')
        :returns:
            The element for this table, containing all the headers and footers and etc
        """
        table_node = etree.Element(
            'Table',
            ID=table_id,
            XFormat="none",
            TableType="OneWay",
            EVFormat="AsteriskAfterNumber")
        title_node = etree.Element('Title')
        title_node.text = self.get_title(column, subset=subset)
        table_node.append(title_node)

        encoder = ColumnEncoder(column)

        # Actually write out the data
        for column_name, column_data in self.group_data(column, subset=subset).items():
            column_node = etree.Element(
                'YColumn',
                Width="70",
                Decimals=f"{encoder.num_decimals:d}",
                Subcolumns="1")

            title_node = etree.Element('Title')
            title_node.text = str(column_name)
            column_node.append(title_node)

            # Write the column data out
            subcolumn_node = etree.Element('Subcolumn')
            for val in column_data:
                val_node = etree.Element('d')
                try:
                    val_node.text = encoder.encode(val)
                except Exception:
                    print(f'Failed to encode value "{val}" in column "{column_name}"')
                    raise
                subcolumn_node.append(val_node)
            column_node.append(subcolumn_node)
            table_node.append(column_node)
        return table_node


class PrismTwoWayTable(PrismTableTemplate):
    """ Store the data for a two-way prism table

    :param DataFrame data:
        The data frame to extract a two-way table from
    :param str xcolumn:
        The column to use for x-data (e.g. timepoint, bin area, etc)
    :param str ycolumn:
        The column to use for y-data (e.g. prism group)
    :param str name_column:
        The column to use for the y-column names (e.g. cohort)
    :param str other_group_column:
        If not None, which column to disaggregate by (e.g. sex)
    :param list[str] order:
        The order to stack the x column in
    """

    def __init__(self,
                 data: pd.DataFrame,
                 xcolumn: str,
                 ycolumn: str,
                 other_group_column: Optional[str] = None,
                 name_column: Optional[str] = None,
                 table_title: Optional[str] = None,
                 order: List[str] = None):
        self.xcolumn = xcolumn
        self.ycolumn = ycolumn
        self.other_group_column = other_group_column

        self.table_title = table_title

        self.name_column = name_column if name_column is not None else ycolumn

        # Normalize the category values
        data = data.copy()
        for column in (xcolumn, ycolumn, other_group_column, name_column):
            if column is None:
                continue
            data.loc[:, column] = data[column].map(clean_group_values)
        self.data = data

        # FIXME: Handle coding here...
        if other_group_column is None:
            self.other_groups = [None]
        else:
            self.other_groups = list(sorted(np.unique(data[other_group_column])))
            if len(self.other_groups) < 2:
                self.other_groups = [None]

        if order is None:
            order = list(sorted(np.unique(data[xcolumn])))
        self.order = order

        self.groups = list(sorted(np.unique(data[ycolumn])))

        self._index = None
        self._num_replicates = None
        self._group_names = None
        self._other_group_index = None

    def group_columns(self):
        """ Work out the mapping from column to group """

        xdata = self.data[self.xcolumn].values
        ydata = self.data[self.ycolumn].values
        name_data = self.data[self.name_column].values

        if self.other_group_column is None:
            other_data = None
        else:
            other_data = self.data[self.other_group_column].values

        # Work out the maximum number of replicates per column
        total_num_replicates = 0
        other_group_index = {}
        group_names = {}
        index = np.zeros((xdata.shape[0], 3), dtype=np.int32)
        for other_group in self.other_groups:
            if other_group is None:
                other_mask = np.ones(ydata.shape, dtype=bool)
            else:
                other_mask = other_data == other_group

            num_replicates = -1
            for j, group in enumerate(self.groups):
                ymask = np.logical_and(ydata == group, other_mask)

                if not np.any(ymask):
                    continue

                names = name_data[ymask]
                if not np.all(names == names[0]):
                    err = f'Got multiple levels for name "{self.name_column}" in group "{group}": {set(names)}'
                    raise ValueError(err)

                group_name = names[0]
                group_names[group] = group_name

                for i, label in enumerate(self.order):
                    mask = np.logical_and(xdata == label, ymask)
                    if not np.any(mask):
                        continue

                    idx, = np.nonzero(mask)
                    got_reps = np.sum(mask)

                    num_replicates = max([num_replicates, got_reps])

                    rep_index = np.arange(got_reps).astype(np.int32)

                    # Rows, cols, replicate
                    assert got_reps <= num_replicates
                    assert len(idx) == len(rep_index)
                    index[idx, 0] = i
                    index[idx, 1] = j
                    index[idx, 2] = rep_index + total_num_replicates

            assert num_replicates > 0

            # Cache the offsets into the replicate stuctures
            other_group_index[other_group] = (
                total_num_replicates,
                total_num_replicates + num_replicates,
            )
            total_num_replicates += num_replicates + 1
        self._index = index
        self._group_names = group_names
        self._other_group_index = other_group_index
        self._num_replicates = total_num_replicates - 1

    def group_data(self, column: str,
                   subset: Optional[str] = None) -> Dict[str, np.ndarray]:
        """ Group data into columns

        :param str column:
            The data column to group this way
        :param str subset:
            If not None, extract this subset of the other groups
        :returns:
            The data, stacked into blocks like a Prism TwoWay table
        """
        values = self.data[column].values
        index = self._index

        if subset is not None:
            st, ed = self._other_group_index.get(subset)
        else:
            st, ed = 0, self._num_replicates

        # Map a given data column into a stack of tables
        tables = {}
        for j, group in enumerate(self.groups):
            group_name = self._group_names[group]
            group_mask = index[:, 1] == j
            group_values = values[group_mask]
            group_index = index[group_mask, :]

            if group_values.dtype == object:
                block = np.full((len(self.order), self._num_replicates), fill_value='', dtype=object)
            else:
                block = np.full((len(self.order), self._num_replicates), fill_value=np.nan)

            block[group_index[:, 0], group_index[:, 2]] = group_values
            tables[str(group_name)] = block[:, st:ed]
        return tables

    def write_table(self, column: str, table_id: str,
                    subset: Optional[str] = None) -> etree.Element:
        """ Write a table out

        :param str column:
            The column to write
        :param str table_id:
            The table ID to write as
        :returns:
            The element for this table, containing all the headers and footers and etc
        """

        # Two way table node
        table_node = etree.Element(
            'Table',
            ID=table_id,
            XFormat="none",
            YFormat="replicates",
            Replicates=str(self._num_replicates),
            TableType="TwoWay",
            EVFormat="AsteriskAfterNumber")

        # Add a title
        title_node = etree.Element('Title')
        title_node.text = self.get_title(column, subset=subset)
        table_node.append(title_node)

        encoder = ColumnEncoder(column)

        # Row titles
        row_title_node = etree.Element('RowTitlesColumn', Width="100")
        sub_row_title_node = etree.Element('Subcolumn')
        for label in self.order:
            node = etree.Element('d')
            node.text = str(label)
            sub_row_title_node.append(node)
        row_title_node.append(sub_row_title_node)
        table_node.append(row_title_node)

        # Columns
        for group_name, block in self.group_data(column, subset=subset).items():
            # Column header
            col_node = etree.Element('YColumn', Width='1000',
                                     Decimals=str(encoder.num_decimals),
                                     Subcolumns=str(self._num_replicates))
            col_title_node = etree.Element('Title')
            col_title_align_node = etree.Element('TextAlign', align='Left')
            col_title_align_node.text = group_name

            col_title_node.append(col_title_align_node)
            col_node.append(col_title_node)

            # Column data
            for j in range(block.shape[1]):
                sub_col_node = etree.Element('Subcolumn')
                for i in range(block.shape[0]):
                    rawval = block[i, j]
                    try:
                        val = encoder.encode(rawval)
                    except Exception:
                        print(f'Failed to encode value "{rawval}" in column "{group_name}"')
                        raise
                    node = etree.Element('d')
                    node.text = val
                    sub_col_node.append(node)
                col_node.append(sub_col_node)
            table_node.append(col_node)
        return table_node


class PrismBuilder(object):
    """ Build a prism PSFX file in memory

    :param datetime now:
        If not None, the datetime object to use for all timestamps
    """

    def __init__(self, now: Optional[datetime.datetime] = None):
        self.root = etree.Element("GraphPadPrismFile", PrismXMLVersion='5.00')

        if now is None:
            now = datetime.datetime.now()
        self.now = now

        # Per-table metadata
        self.tables = {}
        self.table_names = {}
        self.table_cell_type = {}
        self.table_cell_category = {}
        self.table_num_decimals = {}

        self.ref_category_dicts = {
            'sex': {'F': 0, 'Female': 0, 'M': 1, 'Male': 1},
        }

    def write_created_header(self):
        """ Write the created header elements """
        now = self.now.strftime('%Y-%m-%dT%H:%M:%S')

        # Add the created by header
        node = etree.Element('Created')
        node.append(etree.Element(
            'OriginalVersion',
            CreatedByProgram="GraphPad Prism",  # Lies
            CreatedByVersion="9.1.0.216",  # Also lies
            Login="root",
            DateTime=now))
        node.append(etree.Element(
            'MostRecentVersion',
            CreatedByProgram="GraphPad Prism",
            CreatedByVersion="9.1.0.216",
            Login="root",
            DateTime=now))
        self.root.append(node)

    def write_constants(self,
                        data: Dict,
                        root: Optional[etree.Element] = None):
        """ Write constants to the node

        :param dict data:
            A set of key: value pairs
        :param Element root:
            The element tree node to write the constants under
        """
        if root is None:
            root = self.root

        # Write each key,value pair to the dictionary
        for key, val in data.items():
            node = etree.Element('Constant')
            name_node = etree.Element('Name')
            name_node.text = str(key)

            val_node = etree.Element('Value')
            val_node.text = str(val)

            node.append(name_node)
            node.append(val_node)
            root.append(node)

    def write_info_header(self):
        """ Write the info header elements """

        # FIXME: Fake this for now, but add metadata later as needed

        # Add the info sequence header
        node = etree.Element('InfoSequence')
        node.append(etree.Element('Ref', ID='Info0', Selected='1'))
        self.root.append(node)

        # Add the info sequence itself
        node = etree.Element('Info', ID='Info0')
        title_node = etree.Element('Title')
        title_node.text = 'Project info 1'  # FIXME: Autofill this?
        node.append(title_node)

        # FIXME: Autofill this?
        node.append(etree.Element('Notes'))
        self.write_constants({
            'Experiment Date': self.now.strftime('%Y-%m-%d'),
            'Experiment ID': '',
            'Notebook ID': '',
            'Project': '',
            'Experimenter': '',
            'Protocol': '',
        }, root=node)
        self.root.append(node)

    def add_table(self, column: str, template: PrismTableTemplate):
        """ Add a table to the table database

        :param str column:
            The name of the column to add
        :param PrismTableTemplate template:
            The template for this table
        """
        # Include defined subsets too
        for subset in template.total_other_groups:
            table_id = f'Table{len(self.tables):d}'
            self.tables[table_id] = (column, template, subset)

    def write_tables(self):
        """ Write all the tables to the XML object """

        # Write the table index
        node = etree.Element('TableSequence', Selected='1')
        for i, table_id in enumerate(self.tables):
            if i == 0:
                node.append(etree.Element('Ref', ID=table_id, Selected="1"))
            else:
                node.append(etree.Element('Ref', ID=table_id))
        self.root.append(node)

        # Now write out the actual tables
        for table_id, (column, template, subset) in self.tables.items():
            node = template.write_table(column, table_id=table_id, subset=subset)
            self.root.append(node)

    def save_to_file(self, outfile: pathlib.Path):
        """ Actually write the final table out to a file

        :param Path outfile:
            The file to save to
        """
        # Write the final XML to a file
        outfile = pathlib.Path(outfile)
        outfile.parent.mkdir(exist_ok=True, parents=True)
        with outfile.open('wb') as fp:
            fp.write(etree.tostring(self.root, xml_declaration=True, encoding='UTF-8'))


class PrismWriter(object):
    """ Main interface for writing prism files

    .. code-block:: python

        with PrismWriter('./path/to/file.psfx') as writer:
            writer.add_one_way_table(data, group_column='Group', value_column='Value')
            writer.add_two_way_table(data, xcolumn='Day', ycolumn='Group', value_column='Value')

    :param Path outfile:
        Path to write the Prism file to
    """

    def __init__(self, outfile: pathlib.Path):
        self.outfile = pathlib.Path(outfile)
        self.builder = None

    def __enter__(self) -> 'PrismWriter':
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def open(self):
        """ Create the builder object and generate the headers """
        builder = PrismBuilder()
        builder.write_created_header()
        builder.write_info_header()
        self.builder = builder

    def close(self):
        """ Finish writing the file

        .. note:: Unlike with other context manager classes, this class writes upon close

        """
        if self.builder is not None:
            self.builder.write_tables()
            self.builder.save_to_file(self.outfile)
        self.builder = None

    def add_one_way_table(self, data: pd.DataFrame,
                          group_column: str,
                          value_column: str,
                          other_group_column: Optional[str] = None,
                          name_column: Optional[str] = None,
                          subsets: Optional[List[str]] = None,
                          table_title: Optional[str] = None):
        """ Add a one-way table

        :param DataFrame data:
            The pandas data frame with the columns used here
        :param str group_column:
            The column to use to group the data (e.g. Treatment, Animal, etc). Groups
            will be organized in alphabetical order based on this column. If you
            want a different order for the group labels, see ``name_column``
        :param str value_column:
            The column to use for the actual values for each data point
        :param str other_group_column:
            If not None, organize data within the table by a second group (e.g. Sex)
        :param str name_column:
            If not None, use these names as the alias for the columns in ``group_column``
            This lets you make ``group_column`` into a numeric column (e.g. Prism Order)
            then name these columns arbirarily
        :param list[str] subsets:
            If not None, also generate plots for each of these subsets in ``other_group_column``
        :param str table_title:
            If not None, use this for the table title instead of the column name
        """
        table_template = PrismOneWayTable(
            data=data,
            group_column=group_column,
            other_group_column=other_group_column,
            name_column=name_column,
            table_title=table_title)
        table_template.group_columns()

        self.builder.add_table(value_column, table_template)

        # And add any subsets requested
        if subsets is None or other_group_column is None:
            subsets = []
        elif isinstance(subsets, str):
            subsets = [subsets]
        for subset in subsets:
            self.builder.add_table(value_column, table_template, subset=subset)

    def add_two_way_table(self, data: pd.DataFrame,
                          xcolumn: str,
                          ycolumn: str,
                          value_column: str,
                          other_group_column: Optional[str] = None,
                          name_column: Optional[str] = None,
                          order: List[str] = None,
                          subsets: Optional[List[str]] = None,
                          table_title: Optional[str] = None):
        """ Add a two-way table

        :param DataFrame data:
            The pandas data frame with the columns used here
        :param str xcolumn:
            The column to use for the continuous independent variable (e.g. Day, Area, etc.)
        :param str ycolumn:
            The column to use for the group category (e.g. animal, treatment, etc)
        :param str value_column:
            The column to use for the actual values for each data point
        :param str other_group_column:
            If not None, organize data within the table by a second group (e.g. Sex)
        :param str name_column:
            If not None, use these names as the alias for the columns in ``ycolumn``
            This lets you make ``ycolumn`` into a numeric column (e.g. Prism Order)
            then name these columns arbirarily
        :param list[str] order:
            If not None, the order of values in xcolumn to use
        :param list[str] subsets:
            If not None, also generate plots for each of these subsets in ``other_group_column``
        """
        table_template = PrismTwoWayTable(
            data=data, xcolumn=xcolumn,
            ycolumn=ycolumn,
            other_group_column=other_group_column,
            name_column=name_column,
            order=order,
            table_title=table_title,
        )
        table_template.group_columns()
        self.builder.add_table(value_column, table_template)

        # And add any subsets requested
        if subsets is None or other_group_column is None:
            subsets = []
        elif isinstance(subsets, str):
            subsets = [subsets]
        for subset in subsets:
            self.builder.add_table(value_column, table_template, subset=subset)

# Functions


def clean_group_values(val: str) -> str:
    """ Normalize a group value

    :param object val:
        The value to normalize
    :returns:
        The group value, with invisible text, etc removed
    """
    if isinstance(val, (int, np.integer)):
        return int(val)
    elif isinstance(val, (float, np.floating)):
        return float(val)
    val = str(val).strip()
    val = reSPACE.sub(' ', val)
    return val


def normalize_column_name(colname: str) -> str:
    """ Normalize the column name

    :param str colname:
        Name of the column
    :returns:
        The column with all the extraneous stuff removed
    """
    if colname is None:
        return None
    return reNORM.sub('_', colname).lower().strip('_')


def normalize_id_vals(val: str) -> float:
    """ Normalize the ID values """
    if np.issubdtype(type(val), np.integer):
        return float(val)
    if np.issubdtype(type(val), np.floating):
        return float(val)
    val = reNUMBER.sub('.', str(val)).strip('.')
    if val == '':
        return np.nan
    lval = val.split('.', 1)
    if len(lval) == 1:
        return float(lval[0])
    lval, rval = lval
    rval = rval.replace('.', '')
    return float(lval + '.' + rval)

# Main function


def df_to_prism(df: pd.DataFrame,
                outfile: pathlib.Path,
                id_column: str,
                main_group_column: str,
                data_columns: List[str],
                other_group_column: Optional[str] = None,
                name_column: Optional[str] = None):
    """ Convert a data frame to a prism file

    :param DataFrame df:
        The data frame to convert
    :param Path outfile:
        The prism file to write
    :param str id_column:
        The name of the column with the primary ID (e.g. "Animal ID")
    :param str main_group_column:
        The name of the column defining the major groups/columns (e.g. "Prism Group")
    :param list[str] data_columns:
        The list of all valid data columns
    :param str other_group_column:
        The name of the column to disaggregate with (e.g. "Sex")
    :param str name_column:
        The name of the column with the names of each group
    """
    assert id_column is not None
    assert main_group_column is not None

    # Subset the dataframe to be only useful columns
    normalized_columns = {
        normalize_column_name(id_column),
        normalize_column_name(main_group_column),
    }
    columns = [id_column, main_group_column]
    if other_group_column is not None:
        columns.append(other_group_column)
        normalized_columns.add(normalize_column_name(other_group_column))
    if name_column is not None:
        columns.append(name_column)
        normalized_columns.add(normalize_column_name(other_group_column))
    real_data_columns = []
    for col in data_columns:
        if col is None:
            continue
        norm_col = normalize_column_name(col)
        if norm_col in normalized_columns:
            continue
        if col in columns:
            continue
        real_data_columns.append(col)
        columns.append(col)
        normalized_columns.add(norm_col)
    data_columns = real_data_columns

    # Make sure all the columns are in the data
    missing_cols = set(columns) - set(df.columns)
    if len(missing_cols) > 0:
        raise ValueError(f'{len(missing_cols)} columns not found in data: {missing_cols}')

    # Keep only a column subset
    df = df[columns].copy(deep=True)

    # Try to drop any rows where all data are missing
    all_missing = np.sum(df[data_columns].isna().values, axis=1)
    all_missing = all_missing < len(data_columns)
    if np.sum(~all_missing) > 0:
        df = df[all_missing]
        print(f'Dropped: {np.sum(~all_missing)} blank records')

    # Make sure we didn't accidentally drop anything
    some_missing = np.sum(df.isna().values, axis=1) > 0
    if np.sum(some_missing) > 0:
        print(f'Got {np.sum(some_missing)} records with missing data')
        # raise ValueError(f'Got {num_missing} missing values')

    df[main_group_column] = df[main_group_column].map(normalize_id_vals)
    main_group_mask = df[main_group_column].isna().values
    if np.sum(main_group_mask) > 0:
        df = df[~main_group_mask]
        print(f'Dropped: {np.sum(main_group_mask)} blank "{main_group_column}" values')

    # Calculate means over tech reps
    replicate_columns = [id_column, name_column, other_group_column]
    replicate_columns = [c for c in replicate_columns if c is not None]

    old_columns = set(df.columns)
    counts_df = df.groupby(replicate_columns, as_index=False).count()
    counts_df.to_excel(outfile.parent / f'{outfile.stem}_counts.xlsx', index=False)

    df = df.groupby(replicate_columns, as_index=False).mean(numeric_only=True)
    new_columns = set(df.columns)

    # Make sure the groupby didn't burn us
    if new_columns != old_columns:
        print(f'Accidentally created {new_columns - old_columns}')
        print(f'Accidentally destroyed {old_columns - new_columns}')
        raise ValueError(f'Got inconsistent columns, expected {len(old_columns)} got {len(new_columns)}')

    # Check for unique animal ids
    animal_ids = df[id_column]

    # Load the grouping and enforce the uniqueness of animal ids
    if animal_ids.shape != np.unique(animal_ids).shape:
        unique, counts = np.unique(animal_ids, return_counts=True)
        non_unique = unique[counts > 1]
        raise ValueError(f'Got non-unique animal IDs: {non_unique}')

    df.to_excel(outfile.parent / f'{outfile.stem}_means.xlsx', index=False)

    # Columns that make no sense to write to the table
    skip_columns = set()
    if name_column is not None:
        skip_columns.add(name_column)
    skip_columns = {normalize_column_name(c) for c in skip_columns}

    with PrismWriter(outfile) as writer:
        # Write out the tables, organized by individual assay
        for i, col in enumerate(df.columns):
            # Skip unhelpful columns
            if normalize_column_name(col) in skip_columns:
                continue
            writer.add_one_way_table(data=df,
                                     group_column=main_group_column,
                                     other_group_column=other_group_column,
                                     name_column=name_column,
                                     value_column=col)

        # Pivot the data frame for summary linegraphs
        extra_columns = [
            id_column, main_group_column, other_group_column, name_column,
        ]
        grouper = PlaqueGrouper()
        pivot_df = grouper.pivot(df, extra_columns=extra_columns)

        for pivot_col in np.unique(pivot_df['Category']):
            # Skip unhelpful columns
            if normalize_column_name(pivot_col) in skip_columns:
                continue

            # Pull out the linegraph and rename it to the proper column name
            sub_df = pivot_df[pivot_df['Category'] == pivot_col]
            sub_df = sub_df.rename(columns={'Value': pivot_col})

            writer.add_two_way_table(data=sub_df,
                                     xcolumn='PlaqueGroup',
                                     ycolumn=main_group_column,
                                     value_column=pivot_col,
                                     other_group_column=other_group_column,
                                     name_column=name_column,
                                     order=grouper.order)
