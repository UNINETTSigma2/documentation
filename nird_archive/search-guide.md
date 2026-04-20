---
orphan: true
---

(Search Guide)=

# NIRD RDA Search Guide

## How to use this guide

This guide is written for researchers/users who want to find datasets efficiently on [NIRD Research Data Archive](https://archive.sigma2.no/), 
 whether you have a specific dataset in mind or are exploring what exists in a field.

This guide is structured around the situations you are likely to find yourself in:
- You have a broad topic and want to see what exists
- You have too many results and need to narrow down
- You have too few results and need to cast wider
- You know exactly what you want and need to pinpoint it
- You want to reproduce or share a search
Read the sections relevant to your situation. 

![the_search_interface](imgs/figure_1_screenshot_portal_search_interface.png "the search interface")
Figure 1: Screenshot of the archive search interface.

## Getting started: The Default Search

The default/basic search lets you quickly find datasets by entering keywords across all key metadata fields at once.
To perform a basic search:
1. Type one or more keywords into the search bar
2. Press Enter or click the search button
The basic search matches your query against all relevant metadata fields, including title, description, creator, contributor, and more. The results are sorted by relevance by default.

```{note}
 On mixed-case terms and model names:
The default search handles plain text well, but struggles with terms that combine upper and lower case letters with numbers,  for example, the climate model name NorESM2. If you search for NorESM2 without quotes, you may get unexpected or incomplete results. Always wrap such terms in double quotes:
"NorESM2"
"CMIP7"
This forces the system to treat the term as an exact phrase rather than tokenising it.
```
What you see in the results:
Each result card shows the dataset title, the first two lines of its description, the creator name(s), DOI, and file size.
 Click the title to open the full dataset record. Use your browser's back button to return to the results,  your query and any active filters are preserved in the URL.

## Advanced Search

The Advanced Search functionality gives you precise control over which metadata fields to query and how to combine conditions. You can expand or hide Advanced Search by clicking the “Advanced” button. Search syntax example is also available on the advanced search interface.
The Advanced Search interface provides:
- A field selector: choose a specific metadata field from the dropdown (e.g. creator_name, title, release_date)
- A query text box: enter the value you want to search for
- Add condition button: add multiple conditions to your query
- AND / OR buttons: combine conditions with logical operators
- Grouping with parentheses (): group conditions to control evaluation order

Note that you need to click on "Add condition" in order to buid the query. As you add condition the query  is displayed as a string (e.g. creator_name:"Ada Lovelace" AND title:"NorESM") and is also reflected in the URL, making it easy to bookmark or share.

