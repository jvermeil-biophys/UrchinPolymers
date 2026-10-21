dirDst = "D:/MicroscopyData/2023-12-06_3T3-LifeAct_JV/ZSlice_with_cortex/";
imgTitles = getList("image.titles");

for (i = 0; i < imgTitles.length; i++) {
   selectImage(imgTitles[i]);
   run("Duplicate...", "duplicate");
   duplicatedImage = getTitle();
   num = substring(duplicatedImage, 2, 4);
   run("Brightness/Contrast...");
   setMinAndMax(0, 8000);
   run("Apply LUT", "stack");
   run("Scale Bar...", "width=10 height=10 font=20 horizontal bold");
   saveAs("PNG", dirDst + substring(duplicatedImage, 10, duplicatedImage.length - 6) + "_cortexSlice-" + num + ".png");
   //selectWindow(duplicatedImage);
   //close();
}