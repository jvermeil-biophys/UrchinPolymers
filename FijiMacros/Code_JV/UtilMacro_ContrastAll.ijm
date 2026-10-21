imgTitles = getList("image.titles");
for (i = 0; i < imgTitles.length; i++) {
   selectImage(imgTitles[i]);
   run("Brightness/Contrast...");
   run("Enhance Contrast", "saturated=1.0");
   run("Invert", "stack");
   //run("Apply LUT");
   //run("Enhance Contrast", "saturated=1.5");
   //run("In [+]");
   //run("In [+]");
   //run("In [+]");
   //run("In [+]");
   //run("Out [-]");
}