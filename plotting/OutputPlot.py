import numpy as np

class OutputPlot:
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
        self.seriesToolTip = []
        self.xAxis = None


    def setYSeries(self, name, data, tooltipText=""):
        #Set the name
        self.seriesName.append(name)

        #Set the data
        dataText = ""
        for i in range(data.size):
            dataText = dataText + str(data[i]) + ","

        dataText = dataText[0:len(dataText) - 1]
        self.seriesData.append(dataText)

        #Set the tooltip
        self.seriesToolTip.append(tooltipText)

    def setSingleDataPoint(self, name, x, y, toolTipText=""):
        #Set the name
        self.seriesName.append(name)

        #Set the data
        dataText = "[" + str(x) + "," + str(y) + "]"
        self.seriesData.append(dataText)

        #Set the tooltip
        self.seriesToolTip.append("'" + toolTipText + "'")

    def setXSeries(self, data):
        dataText = ""
        for i in range(data.size):
            dataText = dataText + str(data[i]) + ","

        dataText = dataText[0:len(dataText) - 1]
        self.xAxis = dataText

    def createOutput(self):
        # This is where all the html(javascript) file is produced
        self.f.write("\n<script>")
        self.f.write("\n$(function () {")
        self.f.write("\n$('#container').highcharts({")
        self.f.write("\ntitle: {")
        self.f.write("\n    text: '"+self.title+"',")
        self.f.write("\n   x: -20 //center")
        self.f.write("\n},")
        self.f.write("\n subtitle: {")
        self.f.write("\n    text: '"+self.subTitle+"',")
        self.f.write("\n    x: -20")
        self.f.write("\n},")
        self.f.write("\nxAxis: {")
        self.f.write("\ntitle: {")
        self.f.write("\ntext: '"+self.xAxisText+"'")
        self.f.write("\n},")
        self.f.write("\n    categories: ["+self.xAxis+"]")
        self.f.write("\n},")
        self.f.write("\nyAxis: {")
        self.f.write("\ntitle: {")
        self.f.write("\ntext: '"+self.yAxisText+"'")
        self.f.write("\n},")
        self.f.write("\n    plotLines: [{")
        self.f.write("\n        value: 0,")
        self.f.write("\n        width: 1,")
        self.f.write("\n       color: '#808080'")
        self.f.write("\n    }]")
        self.f.write("\n},")
        self.f.write("credits: {")
        self.f.write("    enabled: false")
        self.f.write("},")
        self.f.write("\ntooltip: {")
        self.f.write("\n    valueSuffix: ''")
        self.f.write("\n},")
        self.f.write("\nlegend: {")
        self.f.write("\nlayout: 'vertical',")
        self.f.write("\nalign: 'right',")
        self.f.write("\nverticalAlign: 'middle',")
        self.f.write("\nborderWidth: 0")
        self.f.write("\n},")

        #Concatenate series data
        length = len(self.seriesName)
        seriesData = ""
        for i in range(length):
            seriesData = seriesData + "{name:'" + self.seriesName[i] + "',"
            seriesData = seriesData + "data: [" + self.seriesData[i] + "],"

            if(self.seriesToolTip[i] != ""):
                seriesData = seriesData + "tooltip:{pointFormatter: function (){return " + self.seriesToolTip[i] + ";}}"
            seriesData = seriesData + "},"
        seriesData = seriesData[0:len(seriesData)-1]

        self.f.write("\nseries: ["+seriesData+"]")
        self.f.write("\n});")
        self.f.write("\n});")
        self.f.write("\n</script>")



if __name__ == '__main__':
    # Test the plotting functions
    output = OutputPlot("test.html", "Monthly Average Temperature", "Source: WorldClimate.com", "temperature", "")

    #X-axis
    output.setXSeries(np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']))

    #Series data
    output.setYSeries('Tokyo', np.array([7.0, 6.9, 9.5, 14.5, 18.2, 21.5, 25.2, 26.5, 23.3, 18.3, 13.9, 9.6]))
    output.setYSeries('New York', np.array([-0.2, 0.8, 5.7, 11.3, 17.0, 22.0, 24.8, 24.1, 20.1, 14.1, 8.6, 2.5]))
    output.setYSeries('Berlin', np.array([-0.9, 0.6, 3.5, 8.4, 13.5, 17.0, 18.6, 17.9, 14.3, 9.0, 3.9, 1.0]))
    output.setYSeries('London', np.array([3.9, 4.2, 5.7, 8.5, 11.9, 15.2, 17.0, 16.6, 14.2, 10.3, 6.6, 4.8]))
    output.createOutput()