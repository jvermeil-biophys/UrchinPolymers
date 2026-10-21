SCALE = 15.8; // Pix/um
//rect_h = round(6.3 * SCALE);

dir = getDirectory("Select a Directory")
run("Duplicate...", "duplicate");
run("Enhance Contrast...", "saturated=0.35 normalize process_all");
ny = getHeight();
nx = getWidth();
name = getInfo("image.title");

selectImage(name);
makeLine(0, 0, nx, ny);
run("Reslice [/]...", "output=1.000 slice_count=1");
saveAs("Tiff", dir + "/" + substring(name, 0, name.length - 6) + "_1.tif");
close();

selectImage(name);
makeLine(nx, 0, 0, ny);
run("Reslice [/]...", "output=1.000 slice_count=1");
saveAs("Tiff", dir + "/" + substring(name, 0, name.length - 6) + "_2.tif");
close();

selectImage(name);
makeLine(round(nx/2), 0, round(nx/2), ny);
run("Reslice [/]...", "output=1.000 slice_count=1");
saveAs("Tiff", dir + "/" + substring(name, 0, name.length - 6) + "_3.tif");
close();

selectImage(name);
makeLine(0, round(ny/2), nx, round(ny/2));
run("Reslice [/]...", "output=1.000 slice_count=1");
saveAs("Tiff", dir + "/" + substring(name, 0, name.length - 6) + "_4.tif");
close();

//substring(name, 0, name.length - 2)