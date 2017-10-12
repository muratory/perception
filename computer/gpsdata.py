import pickle
import matplotlib.pyplot as plt
import numpy as np
from commonDeepDriveDefine import *
 
data_pos = open("gps_plot_8_1.txt", "rb")
poslist = pickle.load(data_pos)
data_pos.close()

graph = {
    0:   [ (1, 1) ],
    1:   [ (1, 2) ],
    2:   [ (1, 3) ],
    3:   [ (1, 4) ],
    4:   [ (1, 5) ],
    5:   [ (1, 6) ],
    6:   [ (1, 7) ],
    7:   [ (1, 8) ],
    8:   [ (1, 9) ],
    9:   [ (1, 10) ],
    10:  [ (1, 11) ],
    11:  [ (1, 12) ],
    12:  [ (1, 14) ],
    14:  [ (1, 15) ],
    15:  [ (1, 16), (1, 67) ],
    16:  [ (1, 17), (1, 40) ],
    17:  [ (1, 18) ],
    18:  [ (1, 19) ],
    19:  [ (1, 20) ],
    20:  [ (1, 21) ],
    21:  [ (1, 22) ],
    22:  [ (1, 23) ],
    23:  [ (1, 24) ],
    24:  [ (1, 25) ],
    25:  [ (1, 26) ],
    26:  [ (1, 27) ],
    27:  [ (1, 28) ],
    28:  [ (1, 29) ],
    29:  [ (1, 30) ],
    30:  [ (1, 31) ],
    31:  [ (1, 32) ],
    32:  [ (1, 33) ],
    33:  [ (1, 34) ],
    34:  [ (1, 35) ],
    35:  [ (1, 36) ],
    36:  [ (1, 37) ],
    37:  [ (1, 38) ],
    38:  [ (1, 39) ],
    39:  [ (1, 40) ],
    40:  [ (1, 41) ],
    41:  [ (1, 42) ],
    42:  [ (1, 43) ],
    43:  [ (1, 44) ],
    44:  [ (1, 45) ],
    45:  [ (1, 46) ],
    46:  [ (1, 47) ],
    47:  [ (1, 48) ],
    48:  [ (1, 49) ],
    49:  [ (1, 50) ],
    50:  [ (1, 51) ],
    51:  [ (1, 52) ],
    52:  [ (1, 53) ],
    53:  [ (1, 0) ],
    54:  [ (1, 55) ],
    55: [ (1, 56) ],
    56: [ (1, 57) ],
    57: [ (1, 58) ],
    58: [ (1, 59) ],
    59: [ (1, 60) ],
    60: [ (1, 61) ],
    61:  [ (1, 62) ],
    62:  [ (1, 63) ],
    63:  [ (1, 64) ],
    64:  [ (1, 65) ],
    65:  [ (1, 66), (1, 97) ],
    66:  [ (1, 67), (1, 16) ],
    67:  [ (1, 68) ],
    68:  [ (1, 69) ],
    69:  [ (1, 70) ],
    70:  [ (1, 71) ],
    71:  [ (1, 72) ],
    72:  [ (1, 73) ],
    73:  [ (1, 74) ],
    74:  [ (1, 75) ],
    75:  [ (1, 76) ],
    76:  [ (1, 77) ],
    77:  [ (1, 78) ],
    78:  [ (1, 79) ],
    79:  [ (1, 80) ],
    80:  [ (1, 81) ],
    81:  [ (1, 82) ],
    82:  [ (1, 83) ],
    83:  [ (1, 84) ],
    84:  [ (1, 85) ],
    85:  [ (1, 86) ],
    86:  [ (1, 87) ],
    87:  [ (1, 88) ],
    88:  [ (1, 89) ],
    89:  [ (1, 90) ],
    90:  [ (1, 91) ],
    91:  [ (1, 92) ],
    92:  [ (1, 93) ],
    93:  [ (1, 94) ],
    94:  [ (1, 95) ],
    95:  [ (1, 96) ],
    96:  [ (1, 97) ],
    97:  [ (1, 98) ],
    98:  [ (1, 99) ],
    99:  [ (1, 100) ],
    100:  [ (1, 101) ],
    101:  [ (1, 102) ],
    102:  [ (1, 103) ],
    103:  [ (1, 104) ],
    104:  [ (1, 105) ],
    105:  [ (1, 106) ]
}

def neighbourg(s):
    return graph[s]

if __name__ == '__main__':

    data_pos = open("gps_plot_8_1.txt", "rb")
    poslist = pickle.load(data_pos)
    data_pos.close()

    # Plot GPS
    for i in range(0,len(poslist)):
        if gps2MainRoadLabel([poslist[i][1],poslist[i][2]]) == 'IDLE':
            plt.plot(poslist[i][1],poslist[i][2], "b:o")
        else:
            plt.plot(poslist[i][1],poslist[i][2], "r:o")

    plt.title("GPS Map")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Circle (center_x, center_y) with R = 70
    theta = np.linspace(0, 2*np.pi, 40)
    x = mapCenter_x + straightRadius * np.cos(theta)
    y = mapCenter_Y + straightRadius * np.sin(theta)
    plt.plot(x, y)
    plt.axis("equal")

    # Show the window
    plt.show()


    data_pos = open("gps_plot_8_2.txt", "rb")
    poslist = pickle.load(data_pos)
    data_pos.close()

    # Plot GPS
    for i in range(0,len(poslist)):
        if gps2MainRoadLabel([poslist[i][1],poslist[i][2]]) == 'IDLE':
            plt.plot(poslist[i][1],poslist[i][2], "g:o")
        else:
            plt.plot(poslist[i][1],poslist[i][2], "r:o")
            
    plt.title("GPS Map")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Circle (center_x, center_y) with R = 70
    theta = np.linspace(0, 2*np.pi, 40)
    x = mapCenter_x + straightRadius * np.cos(theta)
    y = mapCenter_Y + straightRadius * np.sin(theta)
    plt.plot(x, y)
    plt.axis("equal")

    # Show the window
    plt.show()
   
