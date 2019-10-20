import re

'''
A config class for tuning configuration
It represents a config file with the following structure:

# This line is a comment
# The file begins with section 0, which defines match parameters
# (the selfplay program, the current directory for the selfplay execution, input pgn file
# for the games, search depth, number of games per match) and optimization hyper parameters
name: test
optdir: C:/Learn/dspsa
selfplay: C:/astra/SelfPlay-dnp.exe
playdir: C:/Learn/dspsa
ipgnfile: C:/astra/open-moves/open-moves.fen
depth: 4
games: 8
laststep: 10
msteps: 10

#alpha: 0.501
#rend: 300

[params]
# Section params defines the parameters to be optimized (can be empty or not appear at all)
# with param name, starting value and scale
epMovingMid:  156, 3
epMovingEnd:  156, 3
epMaterMinor: 1, 1

[weights]
# Section weights defines the weights to be optimized (can be empty or not appear at all)
# with param name, starting mid game value, starting end game value, and scale
kingSafe: 1, 0, 1
kingOpen: 2, 4, 1
kingPlaceCent: 8, 1, 1

'''
class Config:
    # These are acceptable fields in section 0, with their type and maybe default value
    # S is string, I integer and F float
    fields = {
        'name': 'S',
        'method': ('S', 'Bayes'),
        'regressor': ('S', 'GP'),
        'triang': ('S', 'DF'),
        'optdir': ('S', '.'),
        'selfplay': 'S',
        'playdir': ('S', '.'),
        'ipgnfile': 'S',
        'ipgnlen': 'I',
        'depth': ('I', 4),
        'nodes': 'I',
        'games': ('I', 16),
        'laststep': ('F', 0.1),
        'alpha': ('F', 0.501),
        'msteps': ('I', 1000),
        'rend': 'I',
        'save': ('I', 10)
    }

    mandatory_fields = ['name', 'selfplay', 'ipgnfile', 'ipgnlen']

    '''
    A config can be initialized either with a file name or with a dictionary
    When called with a file name, the file will be read and transformed into a dictionary which
    will be used for the config creation
    When called with a dictionary, the dictionary keys will be used as attributes of the
    new created config object
    '''
    def __init__(self, source):
        if type(source) != dict:
            # Called with an open file
            source = Config.readConfig(source)
        # Here source is a dictionary
        for name, val in source.items():
            self.__setattr__(name, val)

    @staticmethod
    def accept_data_type(field_name, field_type):
        if field_type not in ['S', 'I', 'F']:
            raise Exception('Config: wrong field type {} for field {}'.format(field_type, field_name))

    @staticmethod
    def create_defaults():
        values = dict()
        for field_name, field_spec in Config.fields.items():
            if type(field_spec) == str:
                Config.accept_data_type(field_name, field_spec)
                values[field_name] = None
            else:
                field_type, field_ini = field_spec
                Config.accept_data_type(field_name, field_type)
                values[field_name] = field_ini
        return values

    @staticmethod
    def readConfig(cof):
        # Transform the config file to a dictionary
        values = Config.create_defaults()
        seen = set()
        sectionNames = [dict(), dict()]
        section = 0
        lineno = 0
        error = False
        for line in cof:
            lineno += 1
            # split the comment path
            line = re.split('#', line)[0].lstrip().rstrip()
            if len(line) > 0:
                if line == '[params]':
                    section = 1
                elif line == '[weights]':
                    section = 2
                else:
                    parts = re.split(r':\s*', line, 1)
                    name = parts[0]
                    val = parts[1]
                    if section == 0:
                        if name in Config.fields:
                            field_type = Config.fields[name]
                            if type(field_type) == tuple:
                                field_type = field_type[0]
                            if field_type == 'S':
                                values[name] = val
                            elif field_type == 'I':
                                values[name] = int(val)
                            elif field_type == 'F':
                                values[name] = float(val)
                            else:
                                raise Exception('Cannot be here!')
                        else:
                            print('Config error in line {:d}: unknown config name {:s}'.format(lineno, name))
                            error = True
                    else:
                        vals = re.split(r',\s*', val)
                        if len(vals) == section + 1:
                            if name in seen:
                                print('Config error in line {:d}: name {:s} already seen'.format(lineno, name))
                                error = True
                            else:
                                seen.add(name)
                                sectionNames[section-1][name] = [int(v) for v in vals]
                        else:
                            print('Config error in line {:d}: should have {:d} values, it has {:d}'.format(lineno, section+1, len(vals)))
                            error = True
        for mand in Config.mandatory_fields:
            if values[mand] is None:
                print('Config does not define mandatory field "{}"'.format(mand))
                error = True

        if error:
            raise Exception('Config file has errors')

        hasScale = False

        # Collect the eval parameters
        values['pnames'] = []
        values['pinits'] = []
        values['pscale'] = []
        for name, vals in sectionNames[0].items():
            val = vals[0]
            scale = vals[1]
            values['pnames'].append(name)
            values['pinits'].append(val)
            values['pscale'].append(scale)
            if scale != 1:
                hasScale = True

        # Collect the eval weights
        for name, vals in sectionNames[1].items():
            mid = vals[0]
            end = vals[1]
            scale = vals[2]
            values['pnames'].append('mid.' + name)
            values['pinits'].append(mid)
            values['pscale'].append(scale)
            values['pnames'].append('end.' + name)
            values['pinits'].append(end)
            values['pscale'].append(scale)
            if scale != 1:
                hasScale = True

        if not hasScale:
            values['pscale'] = None

        return values
