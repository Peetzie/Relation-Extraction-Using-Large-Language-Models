## Disclaimer REBEL

The dataset has been downloaded and modified. I.e similar to other Relation Extraction dataset i've defined the entity type based on its actual value. 

This has been done by: 

1. Finding unique Qids in the dataset. Creating a list of batched SPARQL requests to fetch the information.
2. This test was done multiple times due to rate limiting.
3. The final 27000 items are either not found or the sparql request:SELECT ?instanceOf WHERE {{

   wd:{q} wdt:P31 ?instanceOf.

   }}

   """ The P31 does refer to properties.
4. Online sparql request finds item. Loal does not return items. Online does not always., i.e i cannot take a stand for 27.000 items.
