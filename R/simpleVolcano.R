#' Plot a volcano plot both static and with plotly and produce a DT::datatable
#' @param DE - a differential expression data frame (eg: toptable output, DESeq2::results ouput, etc.)
#' @param fcColumn a string identifying the column name with logFoldChanges
#' @param pColumn a string identifying the column name with pvalues
#' @param geneCol a string identifying the column name with gene names
#' @param fcMin The FC cutoff to be used (defaults to 0.5)
#' @param pMax The value cutoff to be used (defaults to 0.05)
#' @param labels a vector of genes to be plotted
#' @return a plot
#' @export
#' @examples
###' volcanoPlots(DE, fcColumn, pColumn, geneCol)
simpleVolcano <- function(DE, fcColumn, pColumn, geneCol, fcMin = 0.5, pMax = 0.05, labels = NULL) {

  DE$diffexpressed <- "NO"
  DE$diffexpressed[DE[,fcColumn] > fcMin & DE[,pColumn] < pMax] <- "UP"
  DE$diffexpressed[DE[,fcColumn] < -fcMin & DE[,pColumn] < pMax] <- "DOWN"

  DE$logp <- -log10(DE[,pColumn])
  DE$logfc <- DE[,fcColumn]
  DE$gene <- DE[,geneCol]

  p1 <- ggplot(data = DE, aes(x=logfc, y = logp, col = diffexpressed, label = gene)) +
    geom_point() +
    theme_bw() +
    scale_color_manual(values = c("dodgerblue3", "black", "firebrick3")) +
    geom_vline(xintercept = c(-fcMin, fcMin), col = "gray40", linetype = "dashed") +
    geom_hline(yintercept = -log10(pMax), col = "gray40", linetype = "dashed") +
    theme(legend.position = "none")

  if(!is.null(labels)){
    p1 <- p1 + ggrepel::geom_label_repel(
      label = ifelse(DE$gene %in%
                       labels,
                     DE$gene,
                     NA),
      box.padding = unit(0.25, "lines"),
      hjust = 1, max.overlaps = 100)
  }

  return(p1)
}
