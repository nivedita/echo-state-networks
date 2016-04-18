import numpy as np

class TimeSeriesLineBarPlot:

    def __init__(self, fileName, title, subTitle, yAxisText, type="datetime"):
        self.title = title
        self.subTitle = subTitle
        self.yAxisText = yAxisText
        self.type = type
        self.f = open(fileName, 'w')
        self.f.write('\n<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.8.2/jquery.min.js"></script>')
        self.f.write('\n<script src="https://code.highcharts.com/highcharts.js"></script>')
        self.f.write('\n<script src="https://code.highcharts.com/modules/exporting.js"></script>')
        self.f.write('\n<meta http-equiv="content-type" content="text/html; charset=utf-8"></meta>')
        self.f.write('\n<div id="container" style="min-width: 310px; height: 400px; margin: 0 auto"></div>')

        # Initialization to hold series data
        self.seriesName = []
        self.seriesData = []

    def setSeries(self, name, xData, yData):
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
        self.f.write("\n    zoomType: 'xy'")
        self.f.write("\n},")
        self.f.write("\ntitle: {")
        self.f.write("\n    text: '"+self.title+"'")
        self.f.write("\n},")
        self.f.write("\nsubtitle: {")
        self.f.write("\n    text: '"+self.subTitle+"'")
        self.f.write("\n},")
        if (self.type == "datetime"):
            self.f.write("\n    xAxis: {")
            self.f.write("\n        type: 'datetime'")
            self.f.write("\n    },")
        self.f.write("\nyAxis: [{ // Primary yAxis")
        self.f.write("\n    labels: {")
        self.f.write("\n        style: {")
        self.f.write("\n            color: Highcharts.getOptions().colors[1]")
        self.f.write("\n        }")
        self.f.write("\n    },")
        self.f.write("\n    title: {")
        self.f.write("\n        text: '"+self.yAxisText+"',")
        self.f.write("\n        style: {")
        self.f.write("\n            color: Highcharts.getOptions().colors[1]")
        self.f.write("\n        }")
        self.f.write("\n    }")
        self.f.write("\n}],")


        length = len(self.seriesName)
        seriesData = ""
        for i in range(length):
            seriesData = seriesData + "{name:'" + self.seriesName[i] + " Line ',"
            seriesData = seriesData + "type:'line',"
            seriesData = seriesData + "data: [" + self.seriesData[i] + "],"
            seriesData = seriesData + "marker: {"
            seriesData = seriesData + "lineWidth: 2,"
            seriesData = seriesData + "lineColor: Highcharts.getOptions().colors[3],"
            seriesData = seriesData + "fillColor: 'white'"
            seriesData = seriesData + " }"
            seriesData = seriesData + "},"

        for i in range(length):
            seriesData = seriesData + "{name:'" + self.seriesName[i] + " Column',"
            seriesData = seriesData + "type:'column',"
            seriesData = seriesData + "data: [" + self.seriesData[i] + "],"
            seriesData = seriesData + "},"
        seriesData = seriesData[0:len(seriesData)-1]


        # self.f.write("\nseries: [{")
        # self.f.write("\n    type: 'column',")
        # self.f.write("\n    name: 'Predicted',")
        # self.f.write("\n    data: [3, 2, 1, 3, 4]")
        # self.f.write("\n}, {")
        # self.f.write("\n    type: 'column',")
        # self.f.write("\n    name: 'Actual',")
        # self.f.write("\n    data: [2, 3, 5, 7, 6]")
        # self.f.write("\n}, {")
        # self.f.write("\n    type: 'spline',")
        # self.f.write("\n    name: 'Predicted',")
        # self.f.write("\n    data: [3, 2, 1, 3, 4],")
        # self.f.write("\n    marker: {")
        # self.f.write("\n        lineWidth: 2,")
        # self.f.write("\n        lineColor: Highcharts.getOptions().colors[3],")
        # self.f.write("\n        fillColor: 'white'")
        # self.f.write("\n    }")
        # self.f.write("\n},")
        # self.f.write("\n{")
        # self.f.write("\n    type: 'line',")
        # self.f.write("\n    name: 'Actual',")
        # self.f.write("\n    data: [2, 3, 5, 7, 6],")
        # self.f.write("\n    marker: {")
        # self.f.write("\n        lineWidth: 2,")
        # self.f.write("\n        lineColor: Highcharts.getOptions().colors[3],")
        # self.f.write("\n        fillColor: 'white'")
        # self.f.write("\n    }")
        # self.f.write("\n},")
        # self.f.write("\n]")

        self.f.write("\nseries: ["+seriesData+"]")
        self.f.write("\n});")
        self.f.write("\n});")

        self.f.write("\n</script>")



if __name__ == '__main__':
    # Test the plotting functions
    output = TimeSeriesLineBarPlot("test.html", "Competitor Post Traffic", "Measured in terms of interaction rate", "Interaction Rate")
    output.setSeries("Predicted", np.array(["Date.UTC(2013,5,2)", "Date.UTC(2013,5,3)", "Date.UTC(2013,5,4)"]), np.array([0.7695, 0.7448, 0.7645]))
    output.setSeries("Actual", np.array(["Date.UTC(2013,5,2)", "Date.UTC(2013,5,3)", "Date.UTC(2013,5,4)"]), np.array([0.7, 0.6, 0.86]))
    output.createOutput()