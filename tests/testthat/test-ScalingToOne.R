test_that("Sum of scaled data equals 1", {
  testList = MosaicNMF::getTestData()

  testthat::expect_true(sum(MosaicNMF::scaleDataset(testList[[1]]))==1)
})
