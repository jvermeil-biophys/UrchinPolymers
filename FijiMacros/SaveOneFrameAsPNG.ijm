// Settings
//"F:/IntraCellTracking/26-07-29_FastAcq_NBYolk-Fecondation/Raw/OneFrameAsPng/";
dstDir = "F:/IntraCellTracking/26-08-06_FastAcq_Fec_NB-Yolk_mSG-MT/OneFrameAsPng/";
NoFrame = 0;
strFrame = "" + NoFrame;

// Run
imgTitles = getList("image.titles");
for (i = 0; i < imgTitles.length; i++) {
	frameName = substring(imgTitles[i], 0, imgTitles[i].length - 4);
	selectImage(imgTitles[i]);
	//setSlice(NoFrame);
	run("Duplicate...", "use");
	setMinAndMax(0, 300); // CAREFUL HERE ! ADJUST OR COMMENT
	run("Apply LUT");
	run("Set Scale...", "distance=9.2308 known=1 unit=micron");
	run("Scale Bar...", "width=20 height=10 thickness=8 bold hide");
	saveAs("PNG", dstDir + frameName + "_f" + strFrame + ".png");
}
    

