# -*- mode: python ; coding: utf-8 -*-
import sys
from estimators import *

MODEL, SOURCEDIR = None, None

Estimators = {
    '/languages': EstimatorLanguage,
    '/categories': EstimatorCategory,
    '/threads': EstimatorTopics
}

def template(environ, start_response, model, sourcedir):
    process = model(sourcedir)
    data = process.run().encode('utf-8')
    
    start_response("200 OK", [
        ("Content-Type", "text/plain"),
        ("Content-Length", str(len(data)))
    ])
    
    return iter([data])

class Application(object):
    def __init__(self, routes):
        self.routes = routes

    def not_found(self, environ, start_fn):
        start_fn(
            '404 Not Found',
            [('Content-Type', 'text/plain')]
        )
        return ['404 Not Found']

    def __call__(self, environ, start_fn):
        model, sourcedir = Estimators[environ.get('PATH_INFO')], '../data/'
        handler = self.routes.get(environ.get('PATH_INFO')) or self.not_found
        return handler(environ, start_fn, model, sourcedir)

routes = { i: template for i in Estimators }
app = Application(routes)
