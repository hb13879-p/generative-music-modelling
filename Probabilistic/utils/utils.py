def demidify(num):
    if num == 0:
        return "-"
    root_demap = {0 : 'A', 1 : 'Bb', 2:'B',3:'C',4:'Db',5:'D',6:'Eb',7:'E',8:'F',9:'F#',10:'G',11:'Ab'}
    pitch = (num + 3) % 12
    octave = int(num / 12) - 1
    pitch = root_demap[pitch]
    return pitch + str(octave)
