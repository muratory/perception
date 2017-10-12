import Gnuplot

g = Gnuplot.Gnuplot()
g.title('GPS plot')
g.xlabel('X pos')
g.ylabel('Y pos')
g('set auto x')
g('set xtics format ""')
g('set x2tics')
g('set yrange [2700:0]')
g('set term png')
g('set out "gps_map.png"')


databuff = Gnuplot.File("./gps_map.txt", using='2:3:1 with labels offset 0.5,0.5 notitle axes x2y1')
g.plot(databuff)   
