import numpy as np


class ScatterPlot:

    def __init__(self, fileName, title, subTitle, xAxisText, yAxisText):
        self.title = title
        self.subTitle = subTitle
        self.xAxisText = xAxisText
        self.yAxisText = yAxisText
        self.f = open(fileName, 'w')
        self.f.write('\n<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.8.2/jquery.min.js"></script>')
        self.f.write('\n<script src="https://code.highcharts.com/highcharts.js"></script>')
        self.f.write('\n<script src="https://code.highcharts.com/modules/exporting.js"></script>')
        self.f.write('\n<meta http-equiv="content-type" content="text/html; charset=utf-8"></meta>')
        self.f.write('\n<div id="container" style="min-width: 310px; height: 400px; margin: 0 auto"></div>')

        # Initialization to hold series data
        self.seriesName = []
        self.seriesData = []

    def setSeries(self, name, xData, yData, toolTipText=""):
        # Set the name
        self.seriesName.append(name)

        # Set the data
        dataText = ""
        for i in range(xData.size):
            dataText = dataText + "[" + str(xData[i]) + "," + str(yData[i]) + "],"

        dataText = dataText[0:len(dataText) - 1]
        self.seriesData.append(dataText)


    def createOutput(self):
         # This is where all the html(javascript) file is produced
        self.f.write("\n<script>")
        self.f.write("\n$(function () {")
        self.f.write("\n$('#container').highcharts({")
        self.f.write("\nchart: {")
        self.f.write("\ntype: 'scatter',")
        self.f.write("\nzoomType: 'xy'")
        self.f.write("\n},")
        self.f.write("\ntitle: {")
        self.f.write("\ntext: '"+str(self.title)+"'")
        self.f.write("\n},")
        self.f.write("\nsubtitle: {")
        self.f.write("\ntext: '"+str(self.subTitle)+"'")
        self.f.write("\n},")
        self.f.write("\nxAxis: {")
        self.f.write("\ntitle: {")
        self.f.write("\nenabled: true,")
        self.f.write("\ntext: '"+str(self.xAxisText)+"'")
        self.f.write("\n},")
        self.f.write("\nstartOnTick: true,")
        self.f.write("\nendOnTick: true,")
        self.f.write("\nshowLastLabel: true")
        self.f.write("\n},")
        self.f.write("\nyAxis: {")
        self.f.write("\ntitle: {")
        self.f.write("\ntext: '"+str(self.yAxisText)+"'")
        self.f.write("\n}")
        self.f.write("\n},")
        self.f.write("\nlegend: {")
        self.f.write("\nlayout: 'vertical',")
        self.f.write("\nalign: 'right',")
        self.f.write("\nverticalAlign: 'bottom',")
        self.f.write("\nfloating: true,")
        self.f.write("\nbackgroundColor: (Highcharts.theme && Highcharts.theme.legendBackgroundColor) || '#FFFFFF',")
        self.f.write("\nborderWidth: 1")
        self.f.write("\n},")
        self.f.write("\nplotOptions: {")
        self.f.write("\nscatter: {")
        self.f.write("\nmarker: {")
        self.f.write("\nradius: 8,")
        self.f.write("\nstates: {")
        self.f.write("\nhover: {")
        self.f.write("\nenabled: true,")
        self.f.write("\nlineColor: 'rgb(100,100,100)'")
        self.f.write("\n}")
        self.f.write("\n}")
        self.f.write("\n},")
        self.f.write("\nstates: {")
        self.f.write("\nhover: {")
        self.f.write("\nmarker: {")
        self.f.write("\nenabled: false")
        self.f.write("\n}")
        self.f.write("\n}")
        self.f.write("\n},")
        self.f.write("\ntooltip: {")
        self.f.write("\nheaderFormat: '<b>{series.name}</b><br>',")
        self.f.write("\npointFormat: '{point.x} , {point.y} '")
        self.f.write("\n}")

        self.f.write("\n}")
        self.f.write("\n},")

        # Concatenate series data
        length = len(self.seriesName)
        seriesData = ""
        for i in range(length):
            seriesData = seriesData + "{name:'" + self.seriesName[i] + "',"
            seriesData = seriesData + "data: [" + self.seriesData[i] + "],"

            seriesData = seriesData + "},"
        seriesData = seriesData[0:len(seriesData)-1]

        self.f.write("\nseries: ["+seriesData+"]")
        self.f.write("\n});")
        self.f.write("\n});")

        self.f.write("\n</script>")


if __name__ == '__main__':
    # Test the plotting functions
    output = ScatterPlot("scatter.html", "Network Parameters vs Performance", "Random Graph", "Connectivity", "RMSE")
    output.setSeries("Network size of 100", np.array([0.1,0.3, 0.7,0.9]), np.array([0.2, 0.1, 0.5, 0.6]))
    output.createOutput()
