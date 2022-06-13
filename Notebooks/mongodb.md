---
layout: archive-dates
title: MongoDB
toc: true
---

## Data Description

The data contains the domain TV consumptions in JSON. The collection represents the monthly billing in year 2016 (one document for each client and month). There are 12 files 
in total, one per month. On the below, there is an example JSON file:

### Example of one JSON file

```json
  { 
    "_id" : "MK23180/11/1/2016",
    "billing" : "January 2016",
    "TOTAL" : 16.65,
    "Client": { "customer code": "60/17997520/11T", 
                "DNI": "83383980X", 
                "Name": "Jaime", 
                "Surname": ["Aliaga","Oporto"], 
                "Birth date": "18/02/85", 
                "Phone": 555815828, 
                "Email": "jaime5@may.laim.ucetreseme.kom"
              },
    "contract" : { "contract ID" : "MK23180/11",
                  "start date" : "27/11/14",
                  "end date": "26/11/17", 
                  "product" : { "Reference" : "Free Rider",
                                "monthly fee" : 10,
                                "type" : "pay per content",
                                "cost per content" : 2.5,
                                "zapping" : 5,
                                "cost per minute" : 0,
                                "cost per day" : .95,
                                "promotion" : 5
                              },
                  "address" : "16 Holy Trinity Street, Third floor",
                  "town" : "Rabbitsvalley",
                  "ZIP" : "14550",
                  "country" : "Macedonia"
                },
    "Movies": 
                [ 
                { "Date": "15/01/2016", 
                  "Time": "00:57:00", 
                  "Title": "Resident Evil", 
                  "License": { "Date": "15/01/2016", "Time": "00:00:00"}, 
                  "Details": { 
                               "Year": 2002, 
                               "Country": "UK", 
                               "Color": "yes", 
                               "Aspect ratio": 1.85, 
                               "Content Rating": "R", 
                               "Budget": 33000000, 
                               "Gross": 39532308, 
                               "Director": { "Name": "Paul W.S. Anderson", 
                                             "Facebook likes": 545 }, 
                               "Cast": { "Facebook likes": 17902,
                                         "Stars": [ { "Player": "Colin Salmon",
                                                      "Facebook likes": 766 },
                                                    { "Player": "Jaymes Butler",
                                                      "Facebook likes": 2000 },
                                                    { "Player": "Milla Jovovich",
                                                      "Facebook likes": 14000 }  ]
                                       }, 
                               "Language": "English", 
                               "Genres": ["Action", "Horror", "Sci-Fi"],
                               "Keywords": ["amnesia", "quarantine", "special forces",   
                 "virus", "zombie"],
                               "Faces in poster": 0, 
                               "IMDB score": 6.7, 
                               "IMDB link": 
              "http://www.imdb.com/title/tt0120804/?ref_=fn_tt_tt_1", 
                               "Critic reviews": 226, 
                               "User reviews": 1138, 
                               "Voted users": 198701, 
                               "Facebook likes": 0, 
                               "Duration": 100
                             },
                  "Viewing PCT": 99
                },
                { "Date": "22/01/2016", 
                  "Time": "22:01:00", 
                  "Title": "Whatever It Takes", 
                  "License": { "Date": "22/01/2016", "Time": "00:00:00"}, 
                  "Details": { 
                               "Year": 2000, 
                               "Country": "USA", 
                               "Color": "yes", 
                               "Aspect ratio": 1.85, 
                               "Content Rating": "PG-13", 
                               "Budget": 15000000, 
                               "Gross": 8735529, 
                               "Director": { "Name": "David Raynr", 
                                             "Facebook likes": 9 }, 
                               "Cast": { "Facebook likes": 13524,
                                         "Stars": [ { "Player": "Jodi Lyn O'Keefe",
                                                      "Facebook likes": 897 },
                                                    { "Player": "James Franco",
                                                      "Facebook likes": 11000 },
                                                    { "Player": "Marla Sokoloff",
                                                      "Facebook likes": 612 }  ]
                                       }, 
                               "Language": "English", 
                               "Genres": ["Comedy", "Drama", "Romance"],
                               "Keywords": ["male objectification", "manipulative behavior", 
     "modern day adaptation", "narcissistic woman", 
     "promiscuous woman"],
                               "Faces in poster": 4, 
                               "IMDB score": 5.5, 
                               "IMDB link":
    "http://www.imdb.com/title/tt0202402/?ref_=fn_tt_tt_1", 
                               "Critic reviews": 50, 
                               "User reviews": 89, 
                               "Voted users": 8055, 
                               "Facebook likes": 816, 
                               "Duration": 94
                             },
                  "Viewing PCT": 1
                }
              ],
    "Series": 
               [ 
                { "Date": "09/01/2016", 
                  "Time": "21:04:00", 
                  "Title": "Desperate Housewives", 
                  "Season": 5, 
                  "Episode": 3, 
                  "Avg duration": 45, 
                  "Total Episodes": 24, 
                  "Total Seasons": 8, 
                  "License": { "Date": "09/01/2016", "Time": "00:00:00"}, 
                  "Viewing PCT": 98
                },
                { "Date": "15/01/2016", 
                  "Time": "22:40:00", 
                  "Title": "Desperate Housewives", 
                  "Season": 5, 
                  "Episode": 4, 
                  "Avg duration": 45, 
                  "Total Episodes": 24, 
                  "Total Seasons": 8, 
                  "License": { "Date": "15/01/2016", "Time": "00:00:00"}, 
                  "Viewing PCT": 3
                },
                { "Date": "24/01/2016", 
                  "Time": "16:45:00", 
                  "Title": "Desperate Housewives", 
                  "Season": 5, 
                  "Episode": 5, 
                  "Avg duration": 45, 
                  "Total Episodes": 24, 
                  "Total Seasons": 8, 
                  "License": { "Date": "24/01/2016", "Time": "00:00:00"}, 
                  "Viewing PCT": 1
                },
                { "Date": "12/01/2016", 
                  "Time": "19:50:00", 
                  "Title": "Dr Who", 
                  "Season": 22, 
                  "Episode": 4, 
                  "Avg duration": 45, 
                  "Total Episodes": 13, 
                  "Total Seasons": 36, 
                  "License": { "Date": "12/01/2016", "Time": "00:00:00"}, 
                  "Viewing PCT": 98
                },
                { "Date": "20/01/2016", 
                  "Time": "00:23:00", 
                  "Title": "Dr Who", 
                  "Season": 22, 
                  "Episode": 5, 
                  "Avg duration": 45, 
                  "Total Episodes": 13, 
                  "Total Seasons": 36, 
                  "License": { "Date": "20/01/2016", "Time": "00:00:00"}, 
                  "Viewing PCT": 98
                },
                { "Date": "27/01/2016", 
                  "Time": "21:19:00", 
                  "Title": "Dr Who", 
                  "Season": 22, 
                  "Episode": 6, 
                  "Avg duration": 45, 
                  "Total Episodes": 13, 
                  "Total Seasons": 36, 
                  "License": { "Date": "27/01/2016", "Time": "00:00:00"}, 
                  "Viewing PCT": 98
                },
                { "Date": "19/01/2016", 
                  "Time": "14:29:00", 
                  "Title": "Numb3rs", 
                  "Season": 4, 
                  "Episode": 17, 
                  "Avg duration": 43, 
                  "Total Episodes": 18, 
                  "Total Seasons": 6, 
                  "License": { "Date": "19/01/2016", "Time": "00:00:00"}, 
                  "Viewing PCT": 38
                },
                { "Date": "20/01/2016", 
                  "Time": "08:58:00", 
                  "Title": "Perry Mason", 
                  "Season": 6, 
                  "Episode": 17, 
                  "Avg duration": 60, 
                  "Total Episodes": 28, 
                  "Total Seasons": 9, 
                  "License": { "Date": "20/01/2016", "Time": "00:00:00"}, 
                  "Viewing PCT": 98
                },
                { "Date": "21/01/2016", 
                  "Time": "22:21:00", 
                  "Title": "Perry Mason", 
                  "Season": 6, 
                  "Episode": 18, 
                  "Avg duration": 60, 
                  "Total Episodes": 28, 
                  "Total Seasons": 9, 
                  "License": { "Date": "21/01/2016", "Time": "00:00:00"}, 
                  "Viewing PCT": 100
                }
              ],
    "charge date" : "03/05/17",
    "dump date" : "19/12/15"
  }  
```

### First Query

- Query of getting only the Spanish clients *without* second surname from the data:

```
   db.dump_all.find({"contract.country":"Spain","Client.Surname.1":{$exists:0}}).pretty()
```

- Output:

<img src="/images/mongo1.png?raw=true"/>

### Second Query

- Query of getting the emails which nickname (part of the email at the left of the @ symbol) contains a number:

```
   db.dump_all.find({"Client.Email":{$regex: /.*[0-9].*@.*/ }},{"Client.Email":1,"_id":0})
```

In order to search emails that have numbers of the left side of the @ symbol, we use regex function. As a pattern `/.*[0-9].*@.*/` is searched. 
This means that search any numbers before @ symbol. `.*` means that dot is matching every character except line breaks and `quantifier(*)` match 0 or more of the preceding token.

- Output:

<img src="/images/mongo2.png?raw=true"/>

### Third Query

- Query of the most popular movie genre (the one with more movies released that year):
 
In the code, there are 2 parts. In the 1st part, one collection is created with the genres of the most released year. 
After fining the genres, they were written to another collection. 
In the 2nd step, from this collection, all genres inside of array flatted and each type of genre counted with this phrase. `value: {$sum: 1}`

- Step-1 / Finding the most movies released year:

```
  db.dump_all.aggregate([ 
      {$match: {"Movies.Details.Year": {$exists: true}, "Movies.Details.Genres":{$exists: true}}},
      {$project:{Movies:1,_id:0}},{$unwind: "$Movies"},
      {$project:{"Movies.Details":1}},{$unwind: "$Movies.Details"},
      {$project:{"Movies.Details.Year":1,"Movies.Details.Genres":1}},{$unwind: "$Movies.Details.Year"},
      {$group: {_id: "$Movies.Details.Year",count:{$sum:1},genres: {$push: "$Movies.Details.Genres"}}},
      {$sort: {count:-1}}
  ])
 ```
 
- Step2 / Writing the most released year genres to another collection which is called `most_released_year_genres`.

```
db.most_released_year_genres.aggregate([
    {$project: {genres:1}},
    {$unwind:"$genres"},
    {$group: { _id: "$genres"}},
    {$unwind: "$_id"},
    {$group: { _id: "$_id", value: {$sum: 1} }},
    {$sort: {value:-1}}
    ]).pretty()
```

- Output: 
This resulted the number of Drama with 66.

<img src="/images/mongo3.png?raw=true"/>
 
### Forth Query
 
- Query of how many films has directed and average budget and gross for each director:

In the aggregating pipeline, first of all, the records that movies.details.budget, gross and director name are exist were matched. 
Because these fields will be used. And then in match pipeline, the records that are not blank in director name were matched. 
After match, in project step, first the movies were selected and then unwind the movies. The same steps were done for Movies.Details, Movies.Details.Director, Movies.Details.Director.Name. 
Then the data grouped by id which is Director name, and the records sum belong to the same director which represent their number of movies, movies’ average budget and movies’ average gross. 
After that null records were eliminated by using match function. Finally, the data sorted alphabetically.  

```
  db.dump_all.aggregate(
      {$match: {"Movies.Details.Director.Name":{$exists: true},
               "Movies.Details.Budget":{$exists: true},
               "Movies.Details.Gross":{$exists: true},
               "Movies.Details.Director.Name": {$ne:""},
      }},
      {$project: {"Movies":1}},{$unwind: "$Movies"},
      {$project: {"Movies.Details":1}},{$unwind: "$Movies.Details"},
      {$project: {"Movies.Details.Director":1,"Movies.Details.Budget":1,"Movies.Details.Gross":1}},{$unwind: "$Movies.Details.Director"},
      {$project: {"Movies.Details.Director.Name":1,"Movies.Details.Budget":1,"Movies.Details.Gross":1}},{$unwind: "$Movies.Details.Director.Name"},
      {$group: { _id: "$Movies.Details.Director.Name",
                  num_of_movies:{$sum:1},
                  avg_budget:{$avg: "$Movies.Details.Budget"},
                  avg_gross:{$avg: "$Movies.Details.Gross"}}},
      {$match: {"avg_gross":{$ne:null},"avg_budget":{$ne:null}}},
      {$sort: {_id:1}}).pretty()
```

- Output:

<img src="/images/mongo4.png?raw=true"/>

