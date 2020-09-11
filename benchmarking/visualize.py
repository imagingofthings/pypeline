# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
r = lambda x: random.randint(0,x)

# build a rectangle in axes coords
left, width = .1, .8
bottom, height = .1, .8
right = left + width
top = bottom + height


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

# axes coordinates are 0,0 is bottom left and 1,1 is upper right
frame = patches.Rectangle((left, bottom), width, height,
    fill=False, transform=ax.transAxes, clip_on=False
    )

ax.add_patch(frame)


def drawBox(text, box_left, box_width, row = 0, box_height = 0.1):

	if row == 0:
		color = '#8844FF'
	else:
		red = int(255*box_width/width)
		green = r(255)
		blue =  max(0, min(255, 255-row*50 + r(50)))

		color = '#%02X%02X%02X' % (red, green, blue)


	p = patches.Rectangle((box_left, top - box_height*(1+row)), box_width, box_height,
    facecolor=color, transform=ax.transAxes, clip_on=False, linewidth=1,edgecolor='#000000'
    )

	ax.add_patch(p)

	if box_width > 0.05:
		ax.text(box_left*0.5 + box_width*0.5 + left, top - box_height/2 - box_height*row, text,
		        horizontalalignment='center', verticalalignment='center',
		        fontsize=10, transform=ax.transAxes, wrap=True)


infilename  = "lofar_bootes_ps_timing_timestep500.txt"
infile = open(infilename,"r")
bleft1 = left
bleft2 = left
toptime = 1
for line in infile:
	if ':' not in line: continue
	toks = line.split(':')
	time = toks[-1].strip().split(' ')[0]
	name = toks[0]

	print(line)
	if line[:6] == 'Total:':
		drawBox("{0}: {1}s".format(name,time), left, width, row = 0)
		toptime = float(time)
		continue
	if line[0] != ' ':
		drawBox("{0}: {1}s".format(name,time), bleft1, width*float(time)/toptime, row = 1)
		bleft1 += width*float(time)/toptime
	if line[:2] == '  ' and line[2] != ' ':
		drawBox("{0}: {1}s".format(name,time), bleft2, width*float(time)/toptime, row = 2)
		bleft2 += width*float(time)/toptime







'''
ax.text(left, bottom, 'left top',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)

ax.text(left, bottom, 'left bottom',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes)

ax.text(right, top, 'right bottom',
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes)

ax.text(right, top, 'right top',
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax.transAxes)

ax.text(right, bottom, 'center top',
        horizontalalignment='center',
        verticalalignment='top',
        transform=ax.transAxes)

ax.text(left, 0.5*(bottom+top), 'right center',
        horizontalalignment='right',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

ax.text(left, 0.5*(bottom+top), 'left center',
        horizontalalignment='left',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

ax.text(0.5*(left+right), 0.5*(bottom+top), 'middle',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=20,
        transform=ax.transAxes)

ax.text(right, 0.5*(bottom+top), 'centered',
        horizontalalignment='center',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

ax.text(left, top, 'rotated\nwith newlines',
        horizontalalignment='center',
        verticalalignment='center',
        rotation=45,
        transform=ax.transAxes)
'''

ax.set_axis_off()
plt.show()