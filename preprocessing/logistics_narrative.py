"""Template narrativi per Logistics, basati su tipi e qualificatori OCEL."""
import json
from collections import defaultdict
from functools import lru_cache


# Ogni ruolo richiede sia il tipo dell'oggetto sia il qualificatore della relazione.
TEMPLATES = {
    'Register Customer Order': (
        {'orders': ('Customer Order', 'registered CO')},
        'Registered orders {orders}.'),
    'Create Transport Document': (
        {'documents': ('Transport Document', 'created TD'),
         'orders': ('Customer Order', 'TD created for CO')},
        'Created documents {documents} for orders {orders}.'),
    'Book Vehicles': (
        {'vehicles': ('Vehicle', 'booked VHs'),
         'documents': ('Transport Document', 'VHs booked for TD')},
        'Booked vehicles {vehicles} for documents {documents}.'),
    'Order Empty Containers': (
        {'containers': ('Container', 'CRs ordered'),
         'documents': ('Transport Document', 'ordered for TD')},
        'Ordered empty containers {containers} for documents {documents}.'),
    'Pick Up Empty Container': (
        {'containers': ('Container', 'CR picked')},
        'Picked up empty containers {containers}.'),
    'Collect Goods': (
        {'units': ('Handling Unit', 'HU collected')},
        'Collected units {units}.'),
    'Load Truck': (
        {'units': ('Handling Unit', 'HU loaded'),
         'containers': ('Container', 'CR laded'),
         'trucks': ('Truck', 'TR laded')},
        'Loaded units {units}; containers {containers}; trucks {trucks}.'),
    'Drive to Terminal': (
        {'containers': ('Container', 'CR moved'), 'trucks': ('Truck', 'TR moved')},
        'Moved containers {containers} to terminal on trucks {trucks}.'),
    'Weigh': (
        {'containers': ('Container', 'CR weighted'), 'forklifts': ('Forklift', 'FL weighing')},
        'Weighed containers {containers} using forklifts {forklifts}.'),
    'Place in Stock': (
        {'containers': ('Container', 'CR stored'), 'forklifts': ('Forklift', 'FL moved')},
        'Stored containers {containers} using forklifts {forklifts}.'),
    'Bring to Loading Bay': (
        {'containers': ('Container', 'CR brought to bay'), 'forklifts': ('Forklift', 'FL moved')},
        'Moved containers {containers} to loading bay using forklifts {forklifts}.'),
    'Load to Vehicle': (
        {'containers': ('Container', 'CR loaded'), 'vehicles': ('Vehicle', 'VH laded'),
         'forklifts': ('Forklift', 'FL moved')},
        'Loaded containers {containers} onto vehicles {vehicles} using forklifts {forklifts}.'),
    'Depart': (
        {'vehicles': ('Vehicle', 'VH departed'), 'containers': ('Container', 'CR departed'),
         'documents': ('Transport Document', 'TD with CR departure')},
        'Departed vehicles {vehicles}; containers {containers}; documents {documents}.'),
    'Reschedule Container': (
        {'containers': ('Container', 'CR rescheduled'),
         'documents': ('Transport Document', 'TD with CR rescheduled'),
         'vehicles': ('Vehicle', 'booked VH')},
        'Rescheduled containers {containers}; documents {documents}; booked vehicles {vehicles}.'),
}


def normalize(activity):
    for character in (' ', '+', '-', '_'):
        activity = activity.replace(character, '')
    return activity


# LUPIN normalizza i nomi delle attività prima di costruire i prefissi.
ACTIVITIES = {normalize(name): name for name in TEMPLATES}


def join_ids(ids):
    if len(ids) == 1:
        return ids[0]
    return ', '.join(ids[:-1]) + ' and ' + ids[-1]


@lru_cache(maxsize=50000)
def describe_event(activity, event_objects):
    """Trasforma il JSON di UNA riga CSV in una frase, conservando tutti i ruoli."""
    relations = json.loads(event_objects)
    groups = defaultdict(list)
    for relation in relations:
        key = (relation['type'], relation['qualifier'])
        if relation['id'] not in groups[key]:
            groups[key].append(relation['id'])
    groups = {key: join_ids(sorted(ids)) for key, ids in groups.items()}
    activity = ACTIVITIES.get(normalize(activity), activity)

    if activity in TEMPLATES:
        roles, template = TEMPLATES[activity]
        # Se manca un ruolo o ne compare uno inatteso, descriviamo tutte le relazioni.
        # Non inventiamo oggetti o ruoli e non scartiamo relazioni aggiuntive.
        if set(groups) == set(roles.values()):
            return template.format(**{role: groups[key] for role, key in roles.items()})

    details = '; '.join(
        f'{object_type} {ids} ({qualifier or "unspecified role"})'
        for (object_type, qualifier), ids in sorted(groups.items())
    )
    return f'{activity}: {details}.' if details else f'{activity} (no associated objects).'


class ObjectAliases:
    """Una nuova istanza per traccia; assegna alias alla prima comparsa."""

    PREFIXES = {
        'Customer Order': 'O', 'Transport Document': 'D',
        'Container': 'C', 'Handling Unit': 'H',
        'Truck': 'T', 'Vehicle': 'V', 'Forklift': 'F',
    }

    def __init__(self):
        self.aliases = {}
        self.counts = defaultdict(int)

    def encode(self, event_objects):
        relations = json.loads(event_objects)
        converted = []
        for relation in relations:
            object_id = relation['id']
            object_type = relation['type']
            if object_id not in self.aliases:
                self.counts[object_type] += 1
                self.aliases[object_id] = (
                    self.PREFIXES[object_type] + str(self.counts[object_type])
                )
            # Conserva tipo, qualificatore e qualsiasi altro campo presente.
            converted.append(dict(relation, id=self.aliases[object_id]))
        return json.dumps(converted, ensure_ascii=False)
