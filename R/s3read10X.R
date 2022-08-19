#' Read in 10x output files from an AWS S3 bucket and create a Seurat Object
#'
#' @param bucket The bucket contianing the Cell Ranger outputs (matrix.mtx.gz, features.tsv.gz, and barcodes.tsv.gz)
#' @param region Specify the region for the S3 bucket (usually "us-east-1" for raw data files)
#' @param project must be set, or the function will fail.
#' @param min.cells
#' @param min.features
#' @import aws.s3
#' @import Seurat
#' @import dplyr
#' @import Matrix
#' @return Returns a Seurat object
#' @export

s3.read10X <- function(bucket, region, project = NULL, min.cells = 0, min.features = 0){
  matrix <- aws.s3::s3read_using(FUN = readMM,
                                 bucket = bucket,
                                 object = "matrix.mtx.gz",
                                 opts = list(region = region))


  feature.names <- aws.s3::s3read_using(FUN = read.delim,
                                        bucket = bucket,
                                        object = "features.tsv.gz",
                                        header = FALSE,
                                        stringsAsFactors = FALSE,
                                        opts = list(region = region))
  barcode.names <- aws.s3::s3read_using(FUN = read.delim,
                                        bucket = bucket,
                                        object = "barcodes.tsv.gz",
                                        header = FALSE,
                                        stringsAsFactors = FALSE,
                                        opts = list(region = region))
  colnames(matrix) <- barcode.names$V1
  rownames(matrix) <- feature.names$V2

  if (is.null(project)) {
    object <- Seurat::CreateSeuratObject(counts = matrix,
                                         min.cells = min.cells,
                                         min.features = min.features)
  }

  else {
    object <- Seurat::CreateSeuratObject(counts = matrix,
                                         project = project,
                                         min.cells = min.cells,
                                         min.features = min.features)
  }
}
