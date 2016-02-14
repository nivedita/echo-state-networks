import numpy as np

class CorrelationMatrix:


    def __init__(self, fileName, title, categories):
        self.title = title
        self.categories = categories
        self.f = open(fileName, 'w')
        self.f.write('\n<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.8.2/jquery.min.js"></script>')
        self.f.write('\n<script src="https://code.highcharts.com/highcharts.js"></script>')
        self.f.write('\n<script src="http://code.highcharts.com/modules/heatmap.js"></script>')
        self.f.write('\n<script src="https://code.highcharts.com/modules/exporting.js"></script>')
        self.f.write('\n<meta http-equiv="content-type" content="text/html; charset=utf-8"></meta>')
        self.f.write('\n<div id="container" style="min-width: 310px; height: 400px; margin: 0 auto"></div>')

        # Initialization to hold series data
        self.seriesName = None
        self.seriesData = None

        #Changing categories list to text
        self.categoriesText = ""
        for category in self.categories:
            self.categoriesText = self.categoriesText + "'" + category + "',"
        self.categoriesText = self.categoriesText[0:len(self.categoriesText) - 1]



    def setSeries(self, name, data):
        # Set the name
        self.seriesName = name

        # Set the data
        self.seriesData = data

    def createOutput(self):
        # This is where all the html(javascript) file is produced
        self.f.write("\n<script>")
        self.f.write("\n$(function () {")
        self.f.write("\n$('#container').highcharts({")
        self.f.write("\nchart: {")
        self.f.write("\ntype: 'heatmap',")
        self.f.write("\nmarginTop: 40,")
        self.f.write("\nmarginBottom: 40")
        self.f.write("\n},")
        self.f.write("\ntitle: {")
        self.f.write("\n        text: '"+self.title+"'")
        self.f.write("\n},")
        self.f.write("\nxAxis: {")
        self.f.write("\ncategories: ["+self.categoriesText+"]")
        self.f.write("\n},")
        self.f.write("\nyAxis: {")
        self.f.write("\ncategories: ["+self.categoriesText+"]")
        self.f.write("\n},")
        self.f.write("\ncolorAxis: {")
        self.f.write("\nmin: -1,")
        self.f.write("\nminColor: '#FFFFFF',")
        self.f.write("\nmax: +1,")
        self.f.write("\nmaxColor: '#3AB3CE'")
        self.f.write("\n},")
        self.f.write("\nlegend: {")
        self.f.write("\nalign: 'right',")
        self.f.write("\nlayout: 'vertical',")
        self.f.write("\nmargin: 0,")
        self.f.write("\nverticalAlign: 'top',")
        self.f.write("\ny: 25,")
        self.f.write("\nsymbolHeight: 320")
        self.f.write("\n},")
        self.f.write("\ntooltip: {")
        self.f.write("\nformatter: function () {")
        self.f.write("\nreturn '<b>' + this.series.xAxis.categories[this.point.x] + '</b> vs <br><b>' +")
        self.f.write("\nthis.series.yAxis.categories[this.point.y] + '</b> Correlation: '+")
        self.f.write("\nthis.point.value;")
        self.f.write("\n}")
        self.f.write("\n},")
        self.f.write("\nseries: [{")
        self.f.write("\nname: 'Correlation Matrix',")
        self.f.write("\nborderWidth: 1,")
        self.f.write("\ndata: "+self.seriesData+",")
        self.f.write("\ndataLabels: {")
        self.f.write("\nenabled: true,")
        self.f.write("\ncolor: 'black',")
        self.f.write("\nstyle: {")
        self.f.write("\ntextShadow: 'none'")
        self.f.write("\n}")
        self.f.write("\n}")
        self.f.write("\n}]")
        self.f.write("\n});")
        self.f.write("\n});")
        self.f.write("\n</script>")

if __name__ == '__main__':
    # Test the plotting functions
    categories = ['Fans Change Rate', 'Fans']
    plot = CorrelationMatrix("Correlation.html", "Key Metrics Correlation Matrix", categories)
    plot.setSeries("Correlation Matrix", "[[0,0,1.0],[0,1,0.5],[1,0,-0.5],[1,1,1.0]]")
    plot.createOutput()