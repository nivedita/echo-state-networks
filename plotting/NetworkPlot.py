import numpy as np
from reservoir import ReservoirTopology as topology

class NetworkPlot:
    def __init__(self, fileName, title, networkConnectivityMatrix):
        self.title = title
        self.conn = networkConnectivityMatrix

        self.f = open(fileName, 'w')
        self.data = open("data.json", 'w')

    def __generateHTMLOutput(self):
        self.f.write('\n<!DOCTYPE html>')
        self.f.write('<meta charset="utf-8">')
        self.f.write('<style>')
        self.f.write('.node {')
        self.f.write('stroke: #fff;')
        self.f.write('stroke-width: 1.5px;')
        self.f.write('}')

        self.f.write('.link {')
        self.f.write('stroke: #999;')
        self.f.write('stroke-opacity: .6;')
        self.f.write('}')

        self.f.write('</style>')
        self.f.write('<body>')
        self.f.write('<script src="//d3js.org/d3.v3.min.js"></script>')
        self.f.write('<script>')

        self.f.write('var width = 960,')
        self.f.write('height = 500;')

        self.f.write('var color = d3.scale.category20();')

        self.f.write('var force = d3.layout.force()')
        self.f.write('.charge(-120)')
        self.f.write('.linkDistance(100)')
        self.f.write('.size([width, height]);')

        self.f.write('var svg = d3.select("body").append("svg")')
        self.f.write('.attr("width", width)')
        self.f.write('.attr("height", height);')

        self.f.write('d3.json("data.json", function(error, graph) {')
        self.f.write('if (error) throw error;')

        self.f.write('force')
        self.f.write('.nodes(graph.nodes)')
        self.f.write('.links(graph.links)')
        self.f.write('.start();')

        self.f.write('var link = svg.selectAll(".link")')
        self.f.write('.data(graph.links)')
        self.f.write('.enter().append("line")')
        self.f.write(' .attr("class", "link")')
        self.f.write('.style("stroke-width", function(d) { return Math.sqrt(d.value); });')

        self.f.write(' var node = svg.selectAll(".node")')
        self.f.write('.data(graph.nodes)')
        self.f.write('.enter().append("circle")')
        self.f.write('.attr("class", "node")')
        self.f.write('.attr("r", 10)')
        self.f.write('.style("fill", function(d) { return color(d.group); })')
        self.f.write('.call(force.drag);')

        self.f.write('node.append("title")')
        self.f.write('.text(function(d) { return d.name; });')

        self.f.write('force.on("tick", function() {')
        self.f.write('link.attr("x1", function(d) { return d.source.x; })')
        self.f.write('.attr("y1", function(d) { return d.source.y; })')
        self.f.write('.attr("x2", function(d) { return d.target.x; })')
        self.f.write('.attr("y2", function(d) { return d.target.y; });')

        self.f.write('node.attr("cx", function(d) { return d.x; })')
        self.f.write('.attr("cy", function(d) { return d.y; });')
        self.f.write('});')
        self.f.write('});')

        self.f.write('</script>')

    def __generateDataOutput(self):

        # generate the nodes
        self.data.write('{')
        self.data.write('"nodes":[')
        nodesData = ""
        for i in range(self.conn.shape[0]):
            nodesData += '{"name":"'+str(i)+'","group":1},'
        nodesData = nodesData[0:len(nodesData) - 1]
        self.data.write(nodesData+'],')

        # Generate the links
        self.data.write('"links":[')
        linksData = ""
        for i in range(self.conn.shape[0]):
            for j in range(self.conn.shape[0]):
                if self.conn[i, j] == 1.0 :
                    linksData += '{"source":'+str(i)+',"target":'+str(j)+',"value":1},'

        linksData = linksData[0:len(linksData) - 1]
        self.data.write(linksData+']')
        self.data.write('}')

    def createOutput(self):
        self.__generateHTMLOutput()
        self.__generateDataOutput()

if __name__ == '__main__':
    network2 = topology.ErdosRenyiTopology(size = 10, probability=0.2)
    conn, indices = network2.generateConnectivityMatrix()
    plot2 = NetworkPlot("ErdosRenyi_0.2.html", "", conn)
    plot2.createOutput()

    # network3 = topology.ErdosRenyiTopology(size = 10, probability=0.7)
    # conn, indices = network3.generateConnectivityMatrix()
    # plot3 = NetworkPlot("ErdosRenyi_0.7.html", "", conn)
    # plot3.createOutput()
    #
    # network4 = topology.ErdosRenyiTopology(size = 10, probability=0.5)
    # conn, indices = network4.generateConnectivityMatrix()
    # plot4 = NetworkPlot("ErdosRenyi_0.5.html", "", conn)
    # plot4.createOutput()
