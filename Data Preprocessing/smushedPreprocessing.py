import itertools
from rdflib import Graph
import rdflib as rdf
from urllib.parse import urlparse;
from collections import defaultdict
import hashlib
g = Graph()
graph = g.parse(location='10KSmushedSameAs.ttl', format='ttl')

list_author_list_dict = defaultdict(list)

for e1,r,e2 in graph.triples((None, rdf.term.URIRef('https://schema.org/inAuthorList'), None)):
  list_author_list_dict[e2].append(e1) 
   
for article, authors in zip(list_author_list_dict.keys(),  list_author_list_dict.values()):
  co_authors = []

  #Build list of coauthors: includes all the authors that collaborated on the same paper
  for author in authors:
    print(author)
    co_authors.append(author)

  if len(co_authors) > 1:
    #Add coauthorship relations between authors who worked on the same paper
    for author, co_author in itertools.combinations(co_authors, 2):
        graph.add((author, rdf.term.URIRef("https://schema.org/coAuthorOf"),  co_author))
        graph.add((co_author, rdf.term.URIRef("https://schema.org/coAuthorOf"), author))

'''Add to remove sameAs relations'''
for e1,r,e2 in graph.triples((None, rdf.term.URIRef('http://www.w3.org/2002/07/owl#sameAs'), None)):
  graph.remove((e1,r,e2))
  
'''Add to remove inAuthorList relations'''
for e1,r,e2 in graph.triples((None, rdf.term.URIRef('https://schema.org/inAuthorList'), None)):
  graph.remove((e1,r,e2))

#Write the result to the turtle file updated_results.ttl
graph.serialize(destination="10K_dataset_without_SameAs_final.ttl",format='turtle')
