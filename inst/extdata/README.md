The `inst/extdata` directory stores data that will be installed with the
R package. After installation this directory is available on the local file
system and the path can be retrieved via

```r
system.file("extdata", package = "PACKAGENAME")
```

(Replace `PACKAGENAME` with the name of the parent R package.)