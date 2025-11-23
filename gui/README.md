# GEQIE GUI

## Docker Compose setup

Execute docker commands in the gui directory `cd gui`  
The page will be displayed at `http://localhost:8000`  
The admin panel is available at `http://localhost:8000/admin`  

**Build**
```bash
docker compose build
```

**Start**
```bash
docker compose up
```

**Migrate**
```bash
docker compose exec web python gui/manage.py migrate --noinput
```

**Create superuser**
```bash
docker compose exec web python gui/manage.py createsuperuser
```