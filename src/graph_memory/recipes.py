"""Direct, source-backed recipe collection. No inference or graph exploration."""
import hashlib
import json
from urllib.parse import urlsplit
from .models import Concept,Document,Source,Edge,Relation,stable_id
from .graph import terms

ROOT_ID=stable_id('concept','pakt:recipes')


def validate(body):
    if not isinstance(body,dict) or not isinstance(body.get('name'),str) or not body['name'].strip():raise ValueError('Recipe name required')
    for field in ('ingredients_text','instructions'):
        if not isinstance(body.get(field),list) or not 1<=len(body[field])<=100 or not all(isinstance(x,str) and 0<len(x)<=6000 for x in body[field]):raise ValueError('Invalid recipe content')
    servings=body.get('yield_servings')
    if servings is not None and (type(servings) not in (int,float) or not 0<servings<=60):raise ValueError('Invalid servings')
    return {k:body[k] for k in ('name','ingredients_text','instructions','yield_servings','yield_text','image_url') if k in body}


async def ingest(repository,request):
    body=validate(request.metadata.get('pakt_recipe'));url=request.metadata.get('original_uri','')
    parsed=urlsplit(url)
    if parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password:raise ValueError('Public recipe source required')
    text=json.dumps(body,ensure_ascii=False)
    root=Concept(id=ROOT_ID,label='Recettes',preferred_label='Recettes',aliases=['Recipes'],description='PAKT public recipe collection')
    source=Source(id=stable_id('source','pakt:recipe:'+url),label=body['name'],source_type='url',uri=url,mime_type='text/plain',content_hash=hashlib.sha256(text.encode()).hexdigest(),metadata={'application':'pakt','pakt_recipe':body})
    document=Document(id=stable_id('document','pakt:recipe:'+url),label=body['name'],text=text,document_type='recipe',retrieval_leaf=True,metadata={'original_uri':url,'pakt_recipe':body})
    await repository.put([root,source,document],[Edge(source=source.id,target=document.id,relation=Relation.PROVIDES),Edge(source=document.id,target=ROOT_ID,relation=Relation.ABOUT)])
    return {'source_id':source.id,'document_id':document.id,'collection_id':ROOT_ID}


async def lookup(repository,query='',limit=100):
    # ponytail: bounded direct neighborhood; add scoped pagination if this exceeds 500 recipes.
    nodes,_=await repository.neighbors(ROOT_ID,500)
    wanted=terms(query) if query else set()
    docs=[n for n in nodes if n.kind=='Document' and n.metadata.get('pakt_recipe')]
    docs.sort(key=lambda n:(-len(wanted & terms(n.text)),n.id))
    return {'collection_id':ROOT_ID,'recipes':[{'url':n.metadata['original_uri'],'body':n.metadata['pakt_recipe']} for n in docs[:limit]]}
