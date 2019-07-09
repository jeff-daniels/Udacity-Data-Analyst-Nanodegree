
### Project Submission
# Make Effective Data Visualization
Jeff Daniels  
February 12, 2018

## Summary:

This visualization compares height, weight, body mass index (BMI),
and handedness between Major League Baseball players and American men age
20-29.  It should show that baseball players are much more likely to be
ambidextrous or left-handed and also have a lower BMI.  The visualization also
allows readers to compare the top performing players to the rest of the
players to see if they can draw any conclusions about what makes a great
baseball player.

## Design:

My goal from the beginning was to create an intuitive interface that could
compare the difference in physical characteristics of a sample filtered by
top-performing and bottom-performing athletes.  Exploratory data analysis
using scatterplots and trendlines I felt exagerated weak correlations between
performance and physical characteristics.  I wanted to see what
characteristics were common among players who were much better than average.
Early on I realized I wanted a combination of a scatterplot and some sort of
plot of the distribution of physical characteristics.  I started with
histograms but after receiving feedback realized that box-plots were better at
comparing samples of different sizes.  Eventually I realized that I needed a
graph that represented the proportion of handedness in each sample and so I
added a stacked-normalized-horizontal bar-chart.

I had made an interactive visualization that I was happy with that filtered a
dataset and showed differences in qualitative and categorical data but it was
not telling a story.  I attempted an animated introduction but gave up.  The
text introduction seemed a simple and effective way for readers to get an
overview of the dataset and the buttons hopefully introduced the interactivity
of the visualization.  Eventually I realized that comparing baseball players
to the general population was the best story to tell and added that additional
dataset.

## Feedback:

### Critique 1: Commenting upon two_graphs_v4.html

I found the lack of labels and descriptions on these graphs made it difficult
to interpret the findings at first.  Eventually my focus shifted to that sort
of volume control slider on the scatter plot and found that I could change the
appearance of all the plots by sliding this thing around.  What was I doing if
I was not turning up the bass or the volume?

After the graph was explained to me, the slider filters the scatter plot along
the y-axis and displays the seperate samples below in histograms, I saw that
this could be a nifty little way to explore data.  It was unlike any
combination of graphs I had seen in excel.  You could quickly and intuitively
adjust two groups of data to get an overall sense of how those two groups
differed.  After I learned the graph worked I had a few constructive
suggestions:

  1. Label the axis.  The x-axis on the scatterplot needs to be known as a
  player's physical characteristic.  The y-axis needs to be know as the
  performance we're interested in maximizing.  The histogram y-axis should be
  labelled as count.

  2. The median lines weren't labelled either.
  
  3. The histograms looked cool when you move the slider around but it was
  difficult to see how each group was changing relative to one another when
  one group's size got small.  I think a box plot would be more appropriate.
  A box plot may contain less information, but it would be clearer.
  

### Critique 2: Commenting on three_plots_v4.html

There is a lot of stuff going on in this graph.  I see it as a good reusable
visualization that quickly explores a set of data.  In this simple set of data
you have decided to forgoe drawing trendlines upon the scatter plot in order
to find correlations between player performance and their size and what hand
of theirs is dominant.  Instead, you filter out the top players by performance
and give a statistical snapshot of what these players are like.  It is a
different way of looking at data, which is not necessarily a bad thing.  I
just don't know if the bullet points you present are the best ones for this
visualization.  One intersting point to me was how there is a large group of
players with no batting average or home runs.  Your viz does a good job of
showing how these players make up a large portion of the sample.  They are
noticeably taller.  Are they all pitchers?

Moving the slider up and down and watching the percentage of left-handers
increase is fun.

I think that looking at the top 10% or 1% is more interesting then splitting
the sample in half along the 50th percentile.  Do you bullet point points
still make valid points at higher percentiles?.

It could be nice if you could display the average batting average/home runs
vs. handedness.  I feel like this is the question that most people would want
answered from a visualization.

Mostly, I just don't think you have an interesting story here.  Is there more
to this set of data?

### Critique 3: on three_plots_v4.html

These graphs need a lot of polish.  
I don't like the colors.  
The box plots shouldn't all be one color and none of them should be red on a
white background.  
The handedness legend is weak, lost, and confused.
I didn't know you could click on the paragraphs.  
I guess radio buttons are the right choice for choosing variables, but I hate
the placement.  
I wish I could just enter a percentile directly without fussing with that
button thingy.  
The fonts you chose are lame.  

## Resources:

Good book for learning D3.  
Tooltips using div learned from here.  
Normalized Bar chart reference.  
Interactive Data Visualization for the Web, 2nd Ed. by Scott Murray.

Scatter Plot used a lot of this code as reference:  
http://bl.ocks.org/WilliamQLiu/bd12f73d0b79d70bfbae

Box Plot code used a combination of these plots:  
https://bl.ocks.org/mbostock/4061502
https://bl.ocks.org/rjurney/e04ceddae2e8f85cf3afe4681dac1d74

Bar Plot made using:  
https://bl.ocks.org/mbostock/3886394

General Population Height, weight and BMI data:  
Fryar CD, Gu Q, Ogden CL, Flegal KM. Anthropometric reference data for
children and adults: United States, 2011–2014. National Center for Health
Statistics. Vital Health Stat 3(39). 2016.
https://www.cdc.gov/nchs/data/series/sr_03/sr03_039.pdf

