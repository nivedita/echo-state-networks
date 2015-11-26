import numpy as np

class OutputTimeSeriesPlot:


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
        self.seriesToolTip = []

    def setSeries(self, name, xData, yData, toolTipText=""):
        # Set the name
        self.seriesName.append(name)

        # Set the data
        dataText = ""
        for i in range(xData.size):
            dataText = dataText + "[" + str(xData[i]) + "," + str(yData[i]) + "],"

        dataText = dataText[0:len(dataText) - 1]
        self.seriesData.append(dataText)

        #Set the tooltip
        self.seriesToolTip.append(toolTipText)

    def createOutput(self):
        # This is where all the html(javascript) file is produced
        self.f.write("\n<script>")

        self.f.write("\n$(function () {")
        self.f.write("\n$('#container').highcharts({")
        self.f.write("\n    chart: {")
        self.f.write("\n        zoomType: 'xy'")
        self.f.write("\n    },")
        self.f.write("\n    title: {")
        self.f.write("\n        text: '"+self.title+"'")
        self.f.write("\n    },")
        self.f.write("\n    subtitle: {")
        self.f.write("\n        text: ")
        self.f.write("\n                '"+self.subTitle+"'")
        self.f.write("\n    },")
        if (self.type == "datetime"):
            self.f.write("\n    xAxis: {")
            self.f.write("\n        type: 'datetime'")
            self.f.write("\n    },")
        self.f.write("\n    yAxis: {")
        self.f.write("\n        title: {")
        self.f.write("\n            text: '"+self.yAxisText+"'")
        self.f.write("\n        }")
        self.f.write("\n    },")
        self.f.write("\n    legend: {")
        self.f.write("\n        enabled: true")
        self.f.write("\n    },")
        self.f.write("credits: {")
        self.f.write("    enabled: false")
        self.f.write("},")
        self.f.write("\n    plotOptions: {")
        self.f.write("\n        area: {")
        self.f.write("\n            fillColor: {")
        self.f.write("\n                linearGradient: {")
        self.f.write("\n                    x1: 0,")
        self.f.write("\n                    y1: 0,")
        self.f.write("\n                    x2: 0,")
        self.f.write("\n                    y2: 1")
        self.f.write("\n                },")
        self.f.write("\n                stops: [")
        self.f.write("\n                    [0, Highcharts.getOptions().colors[0]],")
        self.f.write("\n                    [1, Highcharts.Color(Highcharts.getOptions().colors[0]).setOpacity(0).get('rgba')]")
        self.f.write("\n                ]")
        self.f.write("\n            },")
        self.f.write("\n            marker: {")
        self.f.write("\n                radius: 2")
        self.f.write("\n            },")
        self.f.write("\n            lineWidth: 1,")
        self.f.write("\n            states: {")
        self.f.write("\n                hover: {")
        self.f.write("\n                    lineWidth: 1")
        self.f.write("\n                }")
        self.f.write("\n            },")
        self.f.write("\n            threshold: null")
        self.f.write("\n        }")
        self.f.write("\n    },")

         #Concatenate series data
        length = len(self.seriesName)
        seriesData = ""
        for i in range(length):
            seriesData = seriesData + "{name:'" + self.seriesName[i] + "',"
            seriesData = seriesData + "type:'line',"
            seriesData = seriesData + "data: [" + self.seriesData[i] + "],"

            #if(self.seriesToolTip[i] != ""):
                #seriesData = seriesData + "tooltip:{pointFormatter: function (){return " + self.seriesToolTip[i] + ";}}"

            seriesData = seriesData + "},"
        seriesData = seriesData[0:len(seriesData)-1]

        self.f.write("\nseries: ["+seriesData+"]")
        self.f.write("\n});")
        self.f.write("\n});")

        self.f.write("\n</script>")



if __name__ == '__main__':
    # Test the plotting functions
    output = OutputTimeSeriesPlot("test.html", "USD to EUR exchange rate over time", "Click and drag in the plot area to zoom in' : 'Pinch the chart to zoom in", "Exchange rate")
    output.setSeries("USD to EUR", np.array(["Date.UTC(2013,5,2)", "Date.UTC(2013,5,3)", "Date.UTC(2013,5,4)"]), np.array([0.7695, 0.7448, 0.7645]))
    output.createOutput()