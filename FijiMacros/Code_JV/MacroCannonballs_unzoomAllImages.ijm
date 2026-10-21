imgTitles = getList("image.titles");
for (i = 0; i < imgTitles.length; i++) {
   selectImage(imgTitles[i]);
   run("Out [-]");
   //run("Out [-]");
}