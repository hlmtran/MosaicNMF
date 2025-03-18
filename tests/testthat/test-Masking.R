test_that("getMaskMatrix() row and column outputs are equal", {
  testList = MosaicNMF::getTestData()

  missingVectors = MosaicNMF::getMissingIndices(testList)

  mask_r = MosaicNMF::getMaskMatrix(
    missingRows = missingVectors$rowCoord,
    missingColumns = missingVectors$colCoord,
    dim = "row",
    numRow = MosaicNMF:::getTotalRows(testList),
    numCol = length(MosaicNMF:::getUniqueCells(testList))
  )
  mask_c = MosaicNMF::getMaskMatrix(
    missingRows = missingVectors$rowCoord,
    missingColumns = missingVectors$colCoord,
    dim = "col",
    numRow = MosaicNMF:::getTotalRows(testList),
    numCol = length(MosaicNMF:::getUniqueCells(testList))
  )

  testthat::expect_true(identical(mask_c,mask_r))
})

test_that("Mask matrix equal expected value",{
  testList = MosaicNMF::getTestData()

  missingVectors = MosaicNMF::getMissingIndices(testList)

  mask_r = MosaicNMF::getMaskMatrix(
    missingRows = missingVectors$rowCoord,
    missingColumns = missingVectors$colCoord,
    dim = "row",
    numRow = MosaicNMF:::getTotalRows(testList),
    numCol = length(MosaicNMF:::getUniqueCells(testList))
  )
  mask_c = MosaicNMF::getMaskMatrix(
    missingRows = missingVectors$rowCoord,
    missingColumns = missingVectors$colCoord,
    dim = "col",
    numRow = MosaicNMF:::getTotalRows(testList),
    numCol = length(MosaicNMF:::getUniqueCells(testList))
  )

  testthat::expect_true(all(mask_r==is.na(MosaicNMF::stackMatrix(testList,missingValue = NaN))))
})


# mask_r = MosaicNMF::getMaskMatrix(
#   missingRows = missingVectors$rowCoord,
#   missingColumns = missingVectors$colCoord,
#   dim = "row",
#   numRow = MosaicNMF:::getTotalRows(testList),
#   numCol = length(MosaicNMF:::getUniqueCells(testList))
# )
# mask_c = MosaicNMF::getMaskMatrix(
#   missingRows = missingVectors$rowCoord,
#   missingColumns = missingVectors$colCoord,
#   dim = "col",
#   numRow = MosaicNMF:::getTotalRows(testList),
#   numCol = length(MosaicNMF:::getUniqueCells(testList))
# )
# identical(mask_c,mask_r)
# all(mask_r==is.na(MosaicNMF::stackMatrix(testList)))
