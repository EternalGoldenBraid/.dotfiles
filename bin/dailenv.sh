#!/bin/bash

# Setup daily development environment with environmental variables

source ~/Projects/daily/env/bin/activate
export FLASK_APP=daily
export FLASK_ENV=development


cd ~/Projects/daily
