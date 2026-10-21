dirDst = "D:/MicroscopyData/2024-02-27_SpinningDisc_3T3-LifeAct_wPincher_JV/ZStacksForPPT/";
cellId = "C6";
//originalImage = getTitle();

run("Duplicate...", "duplicate");
duplicatedImage = getTitle();
run("Brightness/Contrast...");
setMinAndMax(0, 10000);
run("Apply LUT", "stack");
run("Brightness/Contrast...");

setMinAndMax(0, 65535);
run("Scale Bar...", "width=10 height=10 horizontal bold label");
run("Time Stamper", "starting=0 interval=0.25 x=10 y=20 font=14 decimal=2 anti-aliased or=µm");
run("AVI... ", "compression=JPEG frame=15 save=" + dirDst + cellId + "_ZStack.avi");

Stack.getDimensions(width, height, channels, slices, frames);
setSlice(floor(slices/2));

run("Orthogonal Views");
XZview="XZ " + floor((height)/2);
YZview="YZ " + floor((width)/2);
print(XZview);
print(YZview);
//listeImgs = getList("image.titles");
//for (i = 0; i < listeImgs.length; i++) {
//	print(listeImgs[i]);
//	if (startsWith(listeImgs[i], "XZ")){
//   	   XZview = listeImgs[i];
//    }
//    if (startsWith(listeImgs[i], "YZ")){
//   	   YZview = listeImgs[i];
//    }
//}
selectWindow(XZview);
run("Duplicate...", " ");
run("Rotate 90 Degrees Left");
run("Rotate 90 Degrees Left");
run("Remove Overlay");
run("Scale...", "x=1 y=0.875 width=263 height=140 interpolation=Bilinear average create");
run("Set Scale...", "distance=7.4588 known=1 pixel=1 unit=micron");
run("Scale Bar...", "width=5 height=5 location=[Upper Right] horizontal vertical bold");
saveAs("PNG", dirDst + cellId + "_XZview.png");

selectWindow(YZview);
run("Duplicate...", " ");
run("Rotate 90 Degrees Left");
run("Remove Overlay");
run("Scale...", "x=1 y=0.875 width=263 height=140 interpolation=Bilinear average create");
run("Set Scale...", "distance=7.4588 known=1 pixel=1 unit=micron");
run("Scale Bar...", "width=5 height=5 location=[Upper Right] horizontal vertical bold");
saveAs("PNG", dirDst + cellId + "_YZview.png");

selectWindow(duplicatedImage);
//close();