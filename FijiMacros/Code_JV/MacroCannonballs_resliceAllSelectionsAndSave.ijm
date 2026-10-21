dirDst = "D:/MagneticPincherData/Raw/23.10.28_Cannonballs/M1_M450-2025_BSA_inPBS";
imgTitles = getList("image.titles");
Array.print(imgTitles);
voltage = substring(imgTitles[0], 0, 4);
for (i = 0; i < imgTitles.length; i++) {
   selectImage(imgTitles[i]);
   imageName = substring(imgTitles[i], 0, imgTitles[i].length - 4);
   run("Measure");
   run("Reslice [/]...", "output=1.000 slice_count=1 avoid");
   selectImage("Reslice of " + imageName);
   run("Out [-]");
   run("Out [-]");
   pathDst = dirDst + "/" + "Reslice of " + imageName + ".tif";
   saveAs("Tiff", pathDst);
}
saveAs("Results", dirDst + "/" + voltage + "_TrajAngles.txt");