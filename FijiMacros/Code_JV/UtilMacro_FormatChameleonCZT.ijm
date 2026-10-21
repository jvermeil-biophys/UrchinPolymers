mainDirPath = "D:/MagneticPincherData/Raw/24.06.14_Chameleon/";
fluo_TmpDirPath = mainDirPath + "Tmp_Fluo/";
bf_TmpDirPath = mainDirPath + "Tmp_BF/";
dstDirPath = mainDirPath + "Clean_Fluo_BF/";
ogName = getTitle();

// Channel1 = BF, Channel2 = Fluo

// BF
selectImage(ogName);
run("Duplicate...", "duplicate channels=1");
run("Hyperstack to Stack");
imageSeqPath = bf_TmpDirPath + ogName + "/";
File.makeDirectory(imageSeqPath);
run("Image Sequence... ", "select=" + imageSeqPath + " " + "dir=" + imageSeqPath);
File.openSequence(imageSeqPath);
Image.removeScale();
cleanTifPath = dstDirPath + substring(ogName, 0, ogName.length - 33) + "_BF.tif";
saveAs("Tiff", cleanTifPath);
close();

// Fluo
selectImage(ogName);
run("Duplicate...", "duplicate channels=2");
run("Hyperstack to Stack");
imageSeqPath = fluo_TmpDirPath + ogName + "/";
File.makeDirectory(imageSeqPath);
run("Image Sequence... ", "select=" + imageSeqPath + " " + "dir=" + imageSeqPath);
File.openSequence(imageSeqPath);
Image.removeScale();
cleanTifPath = dstDirPath + substring(ogName, 0, ogName.length - 33) + "_Fluo.tif";
saveAs("Tiff", cleanTifPath);
close();

// folder = getDirectory("Select a directory");
// folder2 = folder + File.separator + "test";
// File.makeDirectory(folder2);