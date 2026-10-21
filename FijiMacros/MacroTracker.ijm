dir = "D://MagneticPincherData//Raw//23.09.19";
name = getInfo("image.title");
resultsName = substring(name, 0, name.length - 4) + "_Results.txt"
run("Analyze Particles...", "size=100-2000 circularity=0.70-1.00 display exclude clear include stack");
saveAs("Results", dir + "/" + resultsName);