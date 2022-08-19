#' Create markdown sub-chunks
#' 
#' @param g Code chunk
#' @param fig.width Scalar numeric, figure width
#' @param fig.height Scalar numeric, figure height
#' @importFrom knitr knit
#' @export 
#' @references http://michaeljw.com/blog/post/subchunkify/
subchunkify <- function(g, fig.height=7, fig.width=5) {
  g_deparsed <- paste0(deparse(
    function() {g}
  ), collapse = '')
  
  sub_chunk <- paste0("
  `","``{r sub_chunk_", basename(tempfile(pattern = "c")), 
                      ", fig.height=", fig.height,
                      ", fig.width=", fig.width, ", echo=FALSE}",
                      "\n(", 
                      g_deparsed
                      , ")()",
                      "\n`","``
  ")
  cat(knitr::knit(text = knitr::knit_expand(text = sub_chunk), quiet = TRUE),
      "\n\n")
}

#' Retrieve a file from AWS S3 and save it locally
#' 
#' @param url Scalar character, URL of a file on AWS S3
#' @param outfile Scalar character, path to a file on the local filesystem
#' @param overwrite Scalar boolean, overwrite `outfile` if it exists?
#' @importFrom aws.s3 get_location save_object
#' @importFrom checkmate assert_flag assert_string test_file_exists
#' @return Invisibly returns `outfile`
#' @export
s3_download <- function(url, outfile = tempfile(), overwrite = FALSE) {
  checkmate::assert_flag(overwrite)
  checkmate::assert_string(url, pattern = "^s3://")
  checkmate::assert_string(outfile)
  if (checkmate::test_file_exists(outfile) & overwrite == FALSE) {
    stop(sprintf("Outfile %s exists and 'overwrite' is FALSE.", outfile), 
         call. = FALSE)
  }
  s3.region <- aws.s3::get_location(url)
  aws.s3::save_object(object = url, file = outfile, region = s3.region)
  return(invisible(outfile))
}


