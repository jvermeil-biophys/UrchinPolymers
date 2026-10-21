from loci.plugins import BF

# parse metadata
from loci.formats import ImageReader
from loci.formats import MetadataTools
reader = ImageReader()
omeMeta = MetadataTools.createOMEXMLMetadata()
reader.setMetadataStore(omeMeta)
file = "D://MagneticPincherData//Raw//24.02.27_Chameleon//Clean//Test//3T3-LifeActGFP_PincherFluo_C7_off4um-01.czi"
reader.setId(file)
seriesCount = reader.getSeriesCount()
for series in range(seriesCount):
    reader.setSeries(series)
    channelCount = omeMeta.getChannelCount(series);
    for channel in range(channelCount):
        channelID = omeMeta.getChannelID(series, channel);
        channelAcqMode = omeMeta.getChannelAcquisitionMode(series, channel);
        channelContrastMeth = omeMeta.getChannelContrastMethod(series, channel);
        
S = reader.getSeriesMetadataValue("Experimenter")
print(S)
        
reader.close()