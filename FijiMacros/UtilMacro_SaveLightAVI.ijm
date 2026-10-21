//dir = "C:/Users/Joseph/Desktop/IntraCellTracking/26-07-29_FastAcq_NBYolk-Fecondation/FilmsLight";
//dir = "C:/Users/Joseph/Desktop/IntraCellTracking/26-06-19_FastAcq_BF/FilmsLight";
dir = "C:/Users/Joseph/Desktop/IntraCellTracking/26-08-06_FastAcq_Fec_NB-Yolk_mSG-MT/FilmsLight";
SCALE = "9.259"; // "4.29"; "9.259";
Nimg = "2000";
FPS = 20;
X_speed = 80;

MULTI_IMG = true;

if (MULTI_IMG) {
	imgTitles = getList("image.titles");
	for (i = 0; i < imgTitles.length; i++) {
		selectImage(imgTitles[i]);
		title = substring(imgTitles[i], 0, imgTitles[i].length - 4);
		path = dir + "/" + title + "_FILM.avi";
		run("Make Substack...", "slices=1-" +Nimg+ "-" + FPS);
		//run("Animation Options...", "speed=80 first=1 last=200");
		run("Enhance Contrast", "saturated=0.35");
		run("Apply LUT", "stack");
		run("Set Scale...", "distance=" + SCALE + " known=1 unit=micron");
		run("Scale Bar...", "width=10 height=5 horizontal bold hide label");
		run("AVI... ", "compression=JPEG frame=" + X_speed + " save=" + path);	
	}
	
} else {
	name = getInfo("image.title");
	path = dir + "/" + substring(name, 0, name.length - 4) + "_FILM.avi";
	run("Make Substack...", "slices=1-" +Nimg+ "-" + FPS);
	//run("Animation Options...", "speed=80 first=1 last=200");
	run("Enhance Contrast", "saturated=0.35");
	run("Apply LUT", "stack");
	run("Set Scale...", "distance=" + SCALE + " known=1 unit=micron");
	run("Scale Bar...", "width=10 height=5 horizontal bold hide label");
	run("AVI... ", "compression=JPEG frame=" + X_speed + " save=" + path);
}	


