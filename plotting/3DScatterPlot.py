import numpy as np


class ScatterPlot3D:

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

    def setSeries(self, name, xData, yData, zData, toolTipText=""):
        # Set the name
        self.seriesName.append(name)

        # Set the data
        dataText = ""
        for i in range(xData.size):
            dataText = dataText + "[" + str(xData[i]) + "," + str(yData[i])+ "," + str(zData[i]) + "],"

        dataText = dataText[0:len(dataText) - 1]
        self.seriesData.append(dataText)


    def createOutput(self):
         # This is where all the html(javascript) file is produced
        self.f.write("\n<script>")
        self.f.write("\n$(function () {")

        self.f.write("\n// Give the points a 3D feel by adding a radial gradient")
        self.f.write("\nHighcharts.getOptions().colors = $.map(Highcharts.getOptions().colors, function (color) {")
        self.f.write("\nreturn {")
        self.f.write("\nradialGradient: {")
        self.f.write("\ncx: 0.4,")
        self.f.write("\ncy: 0.3,")
        self.f.write("\nr: 0.5")
        self.f.write("\n},")
        self.f.write("\nstops: [")
        self.f.write("\n[0, color],")
        self.f.write("\n[1, Highcharts.Color(color).brighten(-0.2).get('rgb')]")
        self.f.write("\n]")
        self.f.write("\n};")
        self.f.write("\n});")
        self.f.write("\n// Set up the chart")
        self.f.write("\nvar chart = new Highcharts.Chart({")
        self.f.write("\nchart: {")
        self.f.write("\nrenderTo: 'container',")
        self.f.write("\nmargin: 100,")
        self.f.write("\ntype: 'scatter',")
        self.f.write("\noptions3d: {")
        self.f.write("\nenabled: true,")
        self.f.write("\nalpha: 10,")
        self.f.write("\nbeta: 30,")
        self.f.write("\ndepth: 250,")
        self.f.write("\nviewDistance: 5,")
        self.f.write("\nfitToPlot: false,")
        self.f.write("\nframe: {")
        self.f.write("\nbottom: { size: 1, color: 'rgba(0,0,0,0.02)' },")
        self.f.write("\nback: { size: 1, color: 'rgba(0,0,0,0.04)' },")
        self.f.write("\nside: { size: 1, color: 'rgba(0,0,0,0.06)' }")
        self.f.write("\n}")
        self.f.write("\n}")
        self.f.write("\n},")
        self.f.write("\ntitle: {")
        self.f.write("\ntext: '"+str(self.title)+"'")
        self.f.write("\n},")
        self.f.write("\nsubtitle: {")
        self.f.write("\ntext: '"+str(self.subTitle)+"'")
        self.f.write("\n},")
        self.f.write("\nplotOptions: {")
        self.f.write("\nscatter: {")
        self.f.write("\nwidth: 10,")
        self.f.write("\nheight: 10,")
        self.f.write("\ndepth: 10")
        self.f.write("\n}")
        self.f.write("\n},")
        self.f.write("\nyAxis: {")
        self.f.write("\nmin: 0,")
        self.f.write("\nmax: 10,")
        self.f.write("\ntitle: null")
        self.f.write("\n},")
        self.f.write("\nxAxis: {")
        self.f.write("\nmin: 0,")
        self.f.write("\nmax: 10,")
        self.f.write("\ngridLineWidth: 1")
        self.f.write("\n},")
        self.f.write("\nzAxis: {")
        self.f.write("\nmin: 0,")
        self.f.write("\nmax: 10,")
        self.f.write("\nshowFirstLabel: false")
        self.f.write("\n},")
        self.f.write("\nlegend: {")
        self.f.write("\nenabled: false")
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
        self.f.write("\n// Add mouse events for rotation")
        self.f.write("\n$(chart.container).bind('mousedown.hc touchstart.hc', function (eStart) {")
        self.f.write("\neStart = chart.pointer.normalize(eStart);")
        self.f.write("\nvar posX = eStart.pageX,")
        self.f.write("\nposY = eStart.pageY,")
        self.f.write("\nalpha = chart.options.chart.options3d.alpha,")
        self.f.write("\nbeta = chart.options.chart.options3d.beta,")
        self.f.write("\nnewAlpha,")
        self.f.write("\nnewBeta,")
        self.f.write("\nsensitivity = 5; // lower is more sensitive")
        self.f.write("\n$(document).bind({")
        self.f.write("\n'mousemove.hc touchdrag.hc': function (e) {")
        self.f.write("\n// Run beta")
        self.f.write("\nnewBeta = beta + (posX - e.pageX) / sensitivity;")
        self.f.write("\nchart.options.chart.options3d.beta = newBeta;")
        self.f.write("\n// Run alpha")
        self.f.write("\nnewAlpha = alpha + (e.pageY - posY) / sensitivity;")
        self.f.write("\nchart.options.chart.options3d.alpha = newAlpha;")
        self.f.write("\nchart.redraw(false);")
        self.f.write("\n},")
        self.f.write("\n'mouseup touchend': function () {")
        self.f.write("\n$(document).unbind('.hc');")
        self.f.write("\n}")
        self.f.write("\n});")
        self.f.write("\n});")
        self.f.write("\n});")
        self.f.write("\n</script>")


if __name__ == '__main__':
    # Test the plotting functions
    output = ScatterPlot3D("scatter3D.html", "Network Parameters vs Performance", "Small World Graphs Graph", "Connectivity", "RMSE")
    output.setSeries("Network size of 100", np.array([0.1,0.3, 0.7,0.9]), np.array([0.2, 0.1, 0.5, 0.6]))
    output.createOutput()
