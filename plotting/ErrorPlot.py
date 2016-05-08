import numpy as np

class ErrorPlot:
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

        #Initialization to hold series data
        self.seriesName = []
        self.seriesData = []
        self.xAxis = None

    def setYAxis(self, name, data):
        #Set the name
        self.seriesName.append(name)

        #Set the data
        dataText = ""
        for i in range(data.size):
            dataText = dataText + str(data[i]) + ","

        dataText = dataText[0:len(dataText) - 1]
        self.seriesData.append(dataText)

    def setXAxis(self, data):
        dataText = ""
        for i in range(data.size):
            dataText = dataText + "'"+str(data[i]) + "',"

        dataText = dataText[0:len(dataText) - 1]
        self.xAxis = dataText

    def createOutput(self):
        # This is where all the html(javascript) file is produced
        self.f.write("\n<script>")
        self.f.write("\n$(function () {")
        self.f.write("\n$('#container').highcharts({")
        self.f.write("\nchart: {")
        self.f.write("\n    type: 'column',")
        self.f.write("\nzoomType: 'xy'")
        self.f.write("\n},")
        self.f.write("\ntitle: {")
        self.f.write("\n    text: '"+self.title+"'")
        self.f.write("\n},")
        self.f.write("\nsubtitle: {")
        self.f.write("\n    text: '"+self.subTitle+"'")
        self.f.write("\n},")
        self.f.write("\nxAxis: {")
        self.f.write("\n   categories: ["+self.xAxis+"],")
        self.f.write("\n    title: {")
        self.f.write("\n        text: '"+self.xAxisText+"'")
        self.f.write("\n    }")
        self.f.write("\n},")
        self.f.write("\nyAxis: {")
        self.f.write("\n    title: {")
        self.f.write("\n        text: '"+self.yAxisText+"',")
        self.f.write("\n        align: 'middle'")
        self.f.write("\n    },")
        self.f.write("\n    labels: {")
        self.f.write("\n        overflow: 'justify'")
        self.f.write("\n    }")
        self.f.write("\n},")
        self.f.write("\ntooltip: {")
        self.f.write("\n    valueSuffix: ' '")
        self.f.write("\n},")
        self.f.write("\nplotOptions: {")
        self.f.write("\n    bar: {")
        self.f.write("\n        dataLabels: {")
        self.f.write("\n            enabled: true")
        self.f.write("\n        }")
        self.f.write("\n    }")
        self.f.write("\n},")
        self.f.write("credits: {")
        self.f.write("    enabled: false")
        self.f.write("},")
        self.f.write("\nlegend: {")
        self.f.write("\n    layout: 'vertical',")
        self.f.write("\n    align: 'right',")
        self.f.write("\n    verticalAlign: 'top',")
        self.f.write("\n    x: -40,")
        self.f.write("\n    y: 80,")
        self.f.write("\n    floating: true,")
        self.f.write("\n    borderWidth: 1,")
        self.f.write("\n    backgroundColor: ((Highcharts.theme && Highcharts.theme.legendBackgroundColor) || '#FFFFFF'),")
        self.f.write("\n    shadow: true")
        self.f.write("\n},")
        self.f.write("\ncredits: {")
        self.f.write("\nenabled: false")
        self.f.write("\n},")


        #Concatenate series data
        length = len(self.seriesName)
        seriesData = ""
        for i in range(length):
            seriesData = seriesData + "{name:'" + self.seriesName[i] + "',"
            seriesData = seriesData + "data: [" + self.seriesData[i] + "]},"
        seriesData = seriesData[0:len(seriesData)-1]

        self.f.write("\nseries: ["+seriesData+"]")
        self.f.write("\n});")
        self.f.write("\n});")
        self.f.write("\n</script>")


if __name__ == '__main__':
    # Test the plotting functions
    output = ErrorPlot("testerror.html", "Comparison of different ESNs", "with different parameters", "ESN Configuration", "Total Error")

    #X-axis
    output.setXAxis(np.array(['With 100', 'With 200', 'With 300']))

    #Series data
    output.setYAxis('Total Regression Error', np.array([107, -31, 635]))
    output.createOutput()