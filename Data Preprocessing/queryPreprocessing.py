from rdflib import Graph
import rdflib as rdf
from collections import defaultdict

g = Graph()
graph = g.parse(location='more_than_10K.ttl', format='ttl')
list_author_list_dict = defaultdict(list)

#Add author lists for sameAs authors
for e1,r,e2 in graph.triples((None, rdf.term.URIRef('http://www.w3.org/2002/07/owl#sameAs'), None)):
  if "contributor/Author" in e2:
    list_id = e2.rsplit(":")[3]
    if (e2, rdf.term.URIRef('https://schema.org/inAuthorList'), rdf.term.URIRef('http://ld.iospress.nl/rdf/contributor/List.Role:authors.ID:' + list_id)) not in graph.triples((None, rdf.term.URIRef('https://schema.org/inAuthorList'), None)):
      print("Added author " + str(e2) + "in list " + str(rdf.term.URIRef('http://ld.iospress.nl/rdf/contributor/List.Role:authors.ID:' + list_id)))
      graph.add((e2, rdf.term.URIRef('https://schema.org/inAuthorList'), rdf.term.URIRef('http://ld.iospress.nl/rdf/contributor/List.Role:authors.ID:' + list_id)))

#remove organizations without a name because not useful for the embeddings
for e1,r,e2 in graph.triples((None, None, None)):
    if "organization/Affiliation" in e1: 
      if graph.value(subject = e1, predicate=rdf.term.URIRef('https://schema.org/hasName')) is None:
        print("Removing organization..." + str(e1))
        graph.remove((e1,r,e2))
    if "organization/Affiliation" in e2: 
      if graph.value(subject = e2, predicate=rdf.term.URIRef('https://schema.org/hasName')) is None:
        print("Removing organization..." + str(e1))
        graph.remove((e1,r,e2))

#Write the result to the turtle file updated_results.ttl
graph.serialize(destination="more_than_10K_preprocessed.ttl",format='turtle')
