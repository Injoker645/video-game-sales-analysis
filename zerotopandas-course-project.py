#!/usr/bin/env python
# coding: utf-8

# # An Analysis On Video Game Sales
# 
# This project is centered around video game sales dataset hosted on kaggle. I'm trying to investigate a few fields of enquiries related to the data. To do so I'll visualize all the relevent data to each field and look for patterns and correlations. The data, which was originally a csv file, is going to be handled as a pandas dataframe. I'll then use matplotlib and seaborn to visualize various graphs. For further analysis I might import other datasets and link them together to find new relations between variables. This project is the result of the 6 week course on data analysis with python provided by jovian.ml in collaboration with freecodecamp. I've learnt about various topics such as using libraries like matplotlib and seaborn to display data, pandas to handle all the data and it's various methods.

# ## Step 1: Data Cleanup
# 
# In this first step we're going to load up the data and try to find any mistakes in the provided data before doing anything else. Since analysis of incorrect data is worse than no analysis, since you confidently believe in your wrong analysis and further use that in other places.
# 
# To begin with we import all the relevant libraries for this notebook and select my graph formatting preferences for later.

# In[1]:


import pandas as pd
import jovian
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
get_ipython().run_line_magic('matplotlib', 'inline')

project_name = "zerotopandas-course-project"

sns.set_style('whitegrid')
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['figure.figsize'] = (10, 5)
matplotlib.rcParams['figure.facecolor'] = '#3080200B'


# Next import the data which is in a csv file located in the current directory so I can directly access it.

# In[2]:


game_sales_original_df = pd.read_csv('vgsales.csv')


# In[3]:


game_sales_original_df


# This dataset has nearly 170000 video game's data recorded and 11 fields of data regarding each game. A small feature to notice is that the games aren't in a randomly indexed order but rather arranged in descending order of overall sales ranking. Let's take a look at what kind of data we have regarding each individual game.
# 
# Note: All the sales are in millions

# In[4]:


game_sales_original_df.columns


# From just looking at the columns, I've decided on three aspects I'll look into. Firstly looking into patterns between regional sales. Secondly correlations between the genres and the year published in. Lastly comparing different games released by the same publisher.

# Now before actually beginning any analysis or graphing, we need to see if we can clean up the data. To do that, we can use in built functions like info and describe, to see an overview of what the data is like.

# In[5]:


game_sales_original_df.info()


# All the data types are correctly assigned with each field and it seems every column is completely filled with the exceptions of Year and Publisher missing a few entries. This is most likely the case for very old games where the publisher doesn't exist anymore and thus information regarding the game is difficult to find. We can investigate the games to see if my guess is true.

# In[6]:


nans_year_df = game_sales_original_df.isnull().any(axis = 1)
game_sales_original_df[nans_year_df].head(30)


# Upon further inspection it seems my hypothesis was wrong. The first part, that the release date is missing for old games, is disproven by the fact that there are games released on the ps3 with missing published dates, meaning they were relatively recently published. Also a quick google search reveals the release date for all of them. The second part of my hypothesis, that the publisher's no longer exist, is disproved by the fact that there well known franchises like WWE Smackdown by THQ or Space Invaders which is a famous revolutionary game by Atari that kickstarted a lot of the gaming industry.
# 
# Now just because our hypothesis was wrong doesn't mean we haven't learnt anything from this inspection. You can take the optimistic approach and say we now know with confidence that the missing values are simply errors made while entering the data, and not because that data is unavailible or unkown.
# 
# 
# We're officially done with step 1

# ## Step 2: Isolating and Analysing Data For Each Inquiry
# 
# Step 2 involves seperating relevent columns and formatting them so that we can analyze the relations between them, without any interference from unrelated data. We'll have to repeat the same procedure for each of the three fields on enquiry we're making.
# 
# 
# 
# ## Enquiry 1: Pattern in regional sales
# 
# I've decided to go two ways about extracting the data. The first method is comparing the regional sales for each individual game as fractions of the total sales of that game. This would tell us which region each game did best in and can later be used.
# Let's try and create that data.
# 
# First we'll just see how the raw sales compare in each region and then add columns representing the fraction of sales for each region since it might be useful later on to arrange the rows using that.

# In[7]:


sales_sum = game_sales_original_df[["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]].sum()
sales_sum


# In[8]:


my_circle=plt.Circle( (0,0), 0.7, color='white')
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.pie(sales_sum, labels=["NA","EU","JP","Other"], labeldistance=0.45, wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' })
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show();


# In[9]:


game_sales_original_df["NA_Sales_Frac"] = game_sales_original_df.NA_Sales/game_sales_original_df.Global_Sales
game_sales_original_df["EU_Sales_Frac"] = game_sales_original_df.EU_Sales/game_sales_original_df.Global_Sales
game_sales_original_df["JP_Sales_Frac"] = game_sales_original_df.JP_Sales/game_sales_original_df.Global_Sales
game_sales_original_df["Others_Sales_Frac"] = game_sales_original_df.Other_Sales/game_sales_original_df.Global_Sales

game_sales_original_df


# Now let's take a look at a few of the games that are sold most in NA.

# In[10]:


top_sold_NA_df = game_sales_original_df.sort_values(["NA_Sales_Frac"], ascending = False)
top_sold_NA_df.head(15)


# It seems that these games were only published in America only and thus all the sales come from that region. Looking at the values of the sales we can see that these were small games, which makes sense as they didn't have the capital to release the game internationally. So we need to remove the games that were only sold in a single region, as their sales fraction values are meaningless to us. We're going to try to accomplish that by removing all the games with sales below a certain threshold.

# In[11]:


int_sales_df = game_sales_original_df[game_sales_original_df.Global_Sales > 0.15]
int_sales_df.tail(15)


# Looking at the lowest sales games it seems like we were successful in reducing the list to only games sold internationally. We've lost some other small games that were sold internationally but even those were largely centered in one region, since small games usually don't have multiple language settings thus limiting it's sales to one language.
# 
# Now let's try to visualize how all the sales are distributed between the regions.

# In[12]:


sales_regions_sum_df = int_sales_df[["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]].sum()
plt.figure(figsize=(12,6))
plt.title("International Games Sales By Region")
plt.pie(sales_regions_sum_df, labels=sales_regions_sum_df.index, autopct='%1.1f%%', startangle=180);


# America takes the lead as expected both due to it's population and the culture surrounding the youth and even adults with respect to video games. What's interesting is that Japan has half as many sales as in all of EU, despite having a much smaller population. This emphazises just how popular it is in that region.

# On a similar tangent, let's try to get which region each game was most popularly sold in and visualize that data.

# In[13]:


conditions = [
    (int_sales_df.NA_Sales > int_sales_df.EU_Sales) & (int_sales_df.NA_Sales > int_sales_df.JP_Sales)
    & (int_sales_df.NA_Sales > int_sales_df.Other_Sales),
    (int_sales_df.EU_Sales > int_sales_df.NA_Sales) & (int_sales_df.EU_Sales > int_sales_df.JP_Sales)
    & (int_sales_df.EU_Sales > int_sales_df.Other_Sales),
    (int_sales_df.JP_Sales > int_sales_df.EU_Sales) & (int_sales_df.JP_Sales > int_sales_df.NA_Sales)
    & (int_sales_df.JP_Sales > int_sales_df.Other_Sales),
    (int_sales_df.Other_Sales > int_sales_df.EU_Sales) & (int_sales_df.Other_Sales > int_sales_df.JP_Sales)
    & (int_sales_df.Other_Sales > int_sales_df.NA_Sales)
]
choices = ["NA", "EU", "JP", "Other"]
pd.options.mode.chained_assignment = None 
int_sales_df["Most_Popular_Game"] = np.select(conditions, choices)
region_count = int_sales_df.Most_Popular_Game.value_counts()
int_sales_df

plt.figure(figsize=(12,6))
plt.title("Most Popular Region In Each Game")
plt.pie(region_count[0:4], labels=["NA", "JP", "EU", "Other"], autopct='%1.1f%%', startangle=180);


# As expected America takes the lead once again by an even larger gap, but the real reason I wanted to display this data is to show that Japan has more titles than Europe. Combining this chart with the previous one, which showed Europe having twice as many sales as Japan, we can confidently infer each regions traits. 
# 
# America is always in the lead due to both it's large population and culcture of gaming. 
# 
# Japan might not have a large population but their culture is even more embedded with video games. That's why they have 16.5% of international titles being sold mostly in Japan. Because due to the demand for games locally, Japan has a thriving video game publishing industry and games aimed at their population. Those games do much better in their own region than anywhere else.
# 
# Europe on the other hand having a much larger population than either of those two regions and the technological advancements to access video games, fall far behind in expected sales. This might be due to the delay in moderness for a lot of the conservative regions. Thus the hobby of video games is largely just contained with the new generation of youth.
# 
# As for the rest of the world, while they do consist for 10% of sales, hardly any large international titles are sold in majority over there. To begin with, we have to realize taking NA, EU and JP out of the picture leaves us with largely third world countries. While the population is much greater than any of the other three regions described above, most places don't have the comfort to have video gaming as a hobby. And most of the sales are just the top selling games which are most likely to be shipped and sold everywhere. As a result it's very unlikely that it would be the most popular region for those games as they are popular everywhere in the world.

# ## Enquiry 2: Identifying Trends Between Genres and Regions

# In[14]:


NA_Highest_Sold = int_sales_df.sort_values("NA_Sales", ascending = False).Name[0]
EU_Highest_Sold = int_sales_df.sort_values("EU_Sales", ascending = False).Name[0]
JP_Highest_Sold = int_sales_df.sort_values("JP_Sales", ascending = False).Name[0]
Other_Highest_Sold = int_sales_df.sort_values("Other_Sales", ascending = False).Name[0]


# In[15]:


NA_Highest_Majority_df = int_sales_df[int_sales_df.Most_Popular_Game == "NA"].copy()
EU_Highest_Majority_df = int_sales_df[int_sales_df.Most_Popular_Game == "EU"]
JP_Highest_Majority_df = int_sales_df[int_sales_df.Most_Popular_Game == "JP"]
Other_Highest_Majority_df = int_sales_df[int_sales_df.Most_Popular_Game == "Other"]

NA_Highest_Majority = NA_Highest_Majority_df.Name[NA_Highest_Majority_df.index[0]]
EU_Highest_Majority = EU_Highest_Majority_df.Name[EU_Highest_Majority_df.index[0]]
JP_Highest_Majority = JP_Highest_Majority_df.Name[JP_Highest_Majority_df.index[0]]
Other_Highest_Majority = Other_Highest_Majority_df.Name[Other_Highest_Majority_df.index[0]]

print("The most sold games were in each region were:\nNA:{0}\nEU:{1}\nJP:{2}\nRest of the world:{3}\nNow for the top sold game that was sold the most in a single region:\nNA:{4}\nEU:{5}\nJP:{6}\nRest of the world:{7}".format(
    NA_Highest_Sold,EU_Highest_Sold,JP_Highest_Sold,Other_Highest_Sold,NA_Highest_Majority,EU_Highest_Majority,
    JP_Highest_Majority,Other_Highest_Majority))


# Wii Sports takes the lead in terms of raw sales for every region due to the huge gap between it and the other games. It has twice as many sales as the game in second place and thus this information doesn't give us much insight. However using the second metric of games with a majority sales in one region we get some new information:
# 
# America takes Wii Sports as expected.
# 
# Europe has Nintendogs, which is pet simulator type game. We could make a hypothesis saying the European region is less interested in action genre games and more in relaxed simulators. We could verify it by checking other games where Europe was the largest sales region.
# 
# Japan has Pokemon Black which makes sense as the pokemon games were born there but the question is why pokemon black which is the 5th generation of pokemon games. Was this the most popular pokemon game or was Japan not the majority of sales for the other ones?
# 
# Lastly GTA: San Andreas was the most popular game that was bought in majority by the rest of the world. We can further investigate how the other games in the GTA series did internationally to see if there's a pattern.
# 
# Let's investigate each of these three issues, starting with the European region. A fair way to see which games genres are popular in Europe are to first filter out the games with a small fraction of sales in Europe. Using the fractions of total sales instead of just sales allows us to see smaller games, as well as blocking out games that weren't popular in Europe but just did so well overall that they still make the cut. Also we're using the international games dataframe for this to work as even unpopular games that are only sold in that region will account for 100% of the sales then.

# In[16]:


EU_Popular_df = int_sales_df[int_sales_df.EU_Sales_Frac > 0.5]
EU_Genre_Count = EU_Popular_df.Genre.value_counts()

EU_Genre_df = EU_Popular_df[["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]].sum()
plt.figure(figsize=(14,8))
plt.title("Most Popular Genres In EU")
plt.pie(EU_Genre_Count, labels=EU_Genre_Count.index, autopct='%1.1f%%', startangle=180);


# Now we can't make assumptions and theories off of just this diagram, without knowing how the genres of international games sold. Now we could make two separate diagrams and make comparisons but an even more efficient method would be creating just one diagram where each genre is given as a fraction of all the games in that genre. Especially since genres like Fighting or Role-Playing could be completely centered in the European region, but this diagram tells us the opposite. Since we're comparing Europe to international sales we might as well add the other regions too. Now in order for all the bars to add up to 100% we'll use all the internationally sold games and divide them into whichever region had the most sales.

# In[17]:


int_genre_count = int_sales_df.Genre.value_counts()
NA_Genre_Count = NA_Highest_Majority_df.Genre.value_counts()
EU_Genre_Count = EU_Highest_Majority_df.Genre.value_counts()
JP_Genre_Count = JP_Highest_Majority_df.Genre.value_counts()
Other_Genre_Count = Other_Highest_Majority_df.Genre.value_counts()
EU_Genre_Count_Perc = EU_Genre_Count*100/int_genre_count
NA_Genre_Count_Perc = NA_Genre_Count*100/int_genre_count
JP_Genre_Count_Perc = JP_Genre_Count*100/int_genre_count
Other_Genre_Count_Perc = Other_Genre_Count*100/int_genre_count
EU_Genre_Count_Perc = EU_Genre_Count_Perc.rename("EU")
NA_Genre_Count_Perc = NA_Genre_Count_Perc.rename("NA")
JP_Genre_Count_Perc = JP_Genre_Count_Perc.rename("JP")
Other_Genre_Count_Perc = Other_Genre_Count_Perc.rename("Other")

final_df = pd.concat([NA_Genre_Count_Perc,EU_Genre_Count_Perc,JP_Genre_Count_Perc,Other_Genre_Count_Perc], axis = 1)
final_df = final_df.T

sns.set(font_scale=1.2)
final_df.set_index(pd.Index(["NA","EU","JP","Other"])).T.plot(kind='bar', stacked=True);


# Now this diagram tells us a different story. The reason all the bars don't go up to 100% is that a few games were tied in sales between regions and thus were discarded.
# 
# But moving on we saw that Sports was the biggest genre in the previous chart. However looking at this one Sports sales in EU are just average compared to their sales in other genres. Racing is better represented genre as it's displayed with high relative values in both charts. It also seems to be Europes largest portion of a genre majority. Role-Playing games in Europe are also reflected well on both charts by having a very small relative portion.
# 
# Now besides Europe, the biggest thing that stands out is Japans domination in the Role-Playing and Strategy genres. Remember Japan is in third place in terms of raw sales so this shows that the sales are really concentrated in a few genres. Looking further into the Role-Playing games genre, I found out that Japan has their own sub genre called JRPG(Japanese Role-Playing Game). This further proves that that the strongest fan-base of RPGs is in Japan.
# 
# A small note is that with this dataset, America is largely dominant in nearly every aspect. But for the genres where others are more successful, we can percieve that in two different ways. One, that the video game genre didn't appeal to the American population as much. Thus other regions would do significantly better even if their sales were relatively average or even lower than their sales in other genres. The second approach is that the genre did average in America, but was very well received by another region and therefore it did much better in sales relative to other genres. In reality it's most likely a combination of both of these reasons to a certain extent.
# 
# Lastly, the rest of the world hardly has any games or genres where they're dominant in sales. This is partly becuase they have the least amount of raw sales in general but even then Japan isn't too far ahead in that regard and yet still manages to have a sizable prescence in several genres. The main reason for the rest of the world not being a majority buyer for games is that the sales are very split between hundreds of countries. There's no shared culture or interest in a certain genre. Most of the sales are just the popular games being shipped everywhere but no chance of the rest of the world to beat America in terms of most sold copies.

# ## Enquiry 3: Pokemon Series
# As we saw previously, Pokemon Black was the highest ranked game sold in majority to Japan. Now Pokemon Black is the fifth generation in it's series and now we need find out why it was ranked first. Was it becuase the other more popular games didn't have a majority of sales in Japan, or is there something special about Pokemon Black giving it first place.

# In[18]:


pokemon_games_df = game_sales_original_df[game_sales_original_df["Name"].str.contains("Pokemon")]
pokemon_games_df


# Looking at all the Pokemon games, we see that Pokemon Black is fifth most sold in the series. First place goes to Pokemon Red which is the oldest one released in 1996. Another thing is that the main game series released every generation are the top sold game with almost all of them having over 10M international sales. The rest of the games are spinoffs/mini series that didn't follow the usual pokemon games style. Now to answer our initial question, we see that the reason Pokemon Black was the first one with majority sales in Japan is because the top 4 games above it are all majority American sales. Now to check the actual popularity of the pokemon game in each region we need to add population data. Population dataframes have a countries column with which you combine the dataframes, but since our data is quite different it's much easier to just hardcode and enter the data ourselves since it's just 4 numbers.
# 
# Note: This data was collected in 2016 so we'll retrieve the population figures for the same year.

# In[20]:


# Population is in millions just like sales
regions = ["NA","EU","JP","Other"]
na_pop = 579
eu_pop = 741.4
jp_pop = 127
others_pop = 7426 - na_pop - eu_pop - jp_pop
pop_arr = [na_pop,eu_pop,jp_pop,others_pop]
regions_sales_sum = game_sales_original_df[["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]].sum()
sales_per_person = regions_sales_sum/pop_arr
sns.barplot(regions, sales_per_person).set_title("Pokemon Games Sold Per Person");
print(regions_sales_sum)


# This diagram finally shows the reality of just how popular the Pokemon franchise is in Japan. With an astounding average of 10 pokemon games sold per person.
# 
# Now obviously this doesn't mean every single person has actually bought 10 pokemon games. There could people who've never bought a single one. But at the same time there are avid fans who've bought every release in the series, which is 35 in this dataset. On the other hand, there are also stores which bought a large stock of the games but might not have sold every last piece. The company would still count these sales, as some third party store owner still bought them. You might also argue the game sold per person argument is invalid as the sales are accounted for a span of 20 years.
# 
# An additional arugement with that would be Japan's unique population change over that same period. The population in 1996 was almost the same as in 2016, off by by under 1%. This is becuase of Japans declining birth rate over this period. The population did go up in the 1996-2016 period but then halfway through started coming back down. Using this knowledge along with the fact that the Pokemon games are aimed at children, although plenty of young adults play it too, we can infer that while the statistic of average games sold per person are technically true, they are inflated due to the aforementioned reasons. Looking at Japan's population demograph through that time period it's true that the percantage of older citizens is increasing but there was a baby boom just a bit before the start of the Pokemon games. That generation grew up with the start of video game technologies like the atari or game boy, and thus it was inevitable that they would come across Pokemon. As they grew older they kept buying the latest addition to the franchise, although of course not all of them would but a large majority as is shown by the data.
# 
# Moving on, North America surprisingly isn't that far behind. With a bit over of 7 pokemon games bought by the average person. However I believe this number is also inflated for another reason. The start of e-commerce with sites like eBay and Amazon meant that anyone could sell anything online to anyone in the world, as long as shipping was handled. While a lot of sales are just within the same country, there are people from countries where the pokemon games might not have been sent or it's out of stock. These people would end up buying from individual sellers, but the game sale would still belong to the original buyers region. However at the end of the day we still need to acknowledge the fact that the Pokemon series is quite popular in the North America region.
# 
# Onto Europe, where it has a lower average of 3 pokemon games per person. There's nothing too intersting about this, as it was expected to have a lower value, as video games in general aren't as popular in Europe as they are in Japan or NA. While it did have twice as many raw sales as Japan, it doesn't even have half as many average sales per person.
# 
# The final bar representing the rest of the world shows us how sparsely video gaming is spread in it, or at least the pokemon video games. While it had around 800 million copies sold which was around 40% lower than Japans sale of 1.3 billion copies, it's average games sold per person is not even 1% of Japans average. This is because we're using the rest of the worlds population to represent the other pokemon sales, but in reality there are plenty of countries that don't even have a single copy bought but their population is still added to it.

# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


game_sales_original_df.Name.unique().size


# In[22]:


game_sales_original_df.Name


# In[23]:


game_sales_original_df[game_sales_original_df.duplicated(subset="Name")]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## References and Future Work
# 
# TODO

# In[ ]:





# Note: The Population for North America was a bit strange to find as I recieved different contradicting answers. This seemed to be because most censuses in the 90s only counted Mexico, Canada and the United States to be part of North America, while the more recent ones inclucde various independant state islands. In the end I decided to just use the three countries to represent NA in 1996, as realistically we can assume they accounted for nearly all the sales.
# 
# North America Population 1996 - http://www.fao.org/3/w4345e/w4345e0i.htm#north%20america1,2 
# 
# Japan Population Demograph 1996 to 2016 - https://www.populationpyramid.net/japan/1996/
# 

# Looking back at splitting the games to one major region, it might've been better to split the games that were tied up equally between all the regions involved. This would both give the representation that the data needed, as well as making our data "complete" meaning all the bars in this diagram would be equal height. The downside would be that it would require longer to process as you'd need compare all the data to find out which ones are tied and then split it up.
