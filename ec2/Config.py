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
# with param name, starting value and scale (name + 2 values)
# For bayesian optimization it should be: param name, minimum, starting and maximum value
# (name + 3 values)
epMovingMid:  156, 3
epMovingEnd:  156, 3
epMaterMinor: 1, 1

[weights]
# Section weights defines the weights to be optimized (can be empty or not appear at all)
# with param name, starting mid game value, starting end game value, and scale (name + 3 values)
# For bayesian optimization it should be: param name, mid minimum, mid starting and mid maximum value,
# end minimum, end starting and end maximum value (name + 6 values)
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
        'acq_func': ('S', 'EI'),    # Could be LCB, EI, gp_hedge and some others
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
        'beta': 'F',
        'msteps': ('I', 1000),  # number of optimizing steps
        'isteps': ('I', 0),     # number of initial steps for GP
        'ropoints': ('I', 5),   # number of restart optimizer points for GP
        'rend': 'I',
        'save': ('I', 10),
        'parallel': ('I', 1),   # number of parallel playing games
        'play_chunk': 'I'       # number of games played in a chunk
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

    # Reading the configuration file
    # There are 3 sections: 0, 1 and 2
    # Section 0 contains general parameter (like optimization method)
    # We have a list with possible field names in this section, and some may have a default
    # Section 1 contains params, i.e. one value per param
    # Section 2 contains weights, i.e. 2 values per weight
    # Number of config values per param/weight depends on optimization method:
    # When using DSPSA, we have the starting params/weights (1/2) plus scale (1)
    # When using Bayes, we have the starting params/weights (1/2) plus inf/sup limit (2/4)
    # All params/weights must have the same number of values
    # So we denote variant 1 as the one with 2/3 values, and variant 2 as the one with 3/6 values

    @staticmethod
    def readConfig(cof):
        # Transform the config file to a dictionary
        values = Config.create_defaults()
        seen = set()
        sectionNames = [dict(), dict()]
        section = 0
        lineno = 0
        error = False
        expect = None
        for line in cof:
            lineno += 1
            # split the comment path
            line = re.split('#', line)[0].lstrip().rstrip()
            if len(line) > 0:
                if line == '[params]':
                    section = 1
                    if expect == 3:
                        expect = 2
                    elif expect == 6:
                        expect = 3
                    else:
                        lenghts = [2, 3]
                elif line == '[weights]':
                    section = 2
                    if expect == 2:
                        expect = 3
                    elif expect == 3:
                        expect = 6
                    else:
                        lenghts = [3, 6]
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
                        if expect is None and len(vals) in lenghts or expect is not None and len(vals) == expect:
                            if expect is None:
                                expect = len(vals)

                            if name in seen:
                                print('Config error in line {:d}: name {:s} already seen'.format(lineno, name))
                                error = True
                            else:
                                seen.add(name)
                                sectionNames[section-1][name] = [int(v) for v in vals]
                        else:
                            if expect is None:
                                print('Config error in line {:d}: should have {:d} or {:d} values, it has {:d}'.format(lineno, section+1, *lenghts, len(vals)))
                            else:
                                print('Config error in line {:d}: should have {:d} values, it has {:d}'.format(lineno, expect, len(vals)))
                            error = True
        for mand in Config.mandatory_fields:
            if values[mand] is None:
                print('Config does not define mandatory field "{}"'.format(mand))
                error = True

        if expect is None:
            print('Config does not have parameters or weights')
            error = True

        if error:
            raise Exception('Config file has errors')

        hasScale = False

        # Collect the eval parameters
        variant = None
        values['pnames'] = []
        values['pinits'] = []
        values['pscale'] = []
        values['pmin'] = []
        values['pmax'] = []
        for name, vals in sectionNames[0].items():
            assert variant is None or len(vals) == 2 and variant == 1 or len(vals) == 3 and variant == 2
            if len(vals) == 2:
                if variant is None:
                    variant = 1
                # We have: name: init, scale
                values['pnames'].append(name)
                values['pinits'].append(vals[0])
                values['pscale'].append(vals[1])
                if scale != 1:
                    hasScale = True
            else:
                if variant is None:
                    variant = 2
                # We have: name: init, min, max
                values['pnames'].append(name)
                values['pinits'].append(vals[0])
                values['pmin'].append(vals[1])
                values['pmax'].append(vals[2])

        # Collect the eval weights
        for name, vals in sectionNames[1].items():
            assert variant is None or len(vals) == 3 and variant == 1 or len(vals) == 6 and variant == 2
            if len(vals) == 3:
                if variant is None:
                    variant = 1
                # We have: name: initMid, initEnd, scale
                values['pnames'].append('mid.' + name)
                values['pinits'].append(vals[0])
                values['pscale'].append(scale)
                values['pnames'].append('end.' + name)
                values['pinits'].append(vals[1])
                values['pscale'].append(vals[2])
                if scale != 1:
                    hasScale = True
            else:
                if variant is None:
                    variant = 2
                # We have: name: initMid, minMid, maxMid, initEnd, minEnd, maxEnd
                values['pnames'].append('mid.' + name)
                values['pinits'].append(vals[0])
                values['pmin'].append(vals[1])
                values['pmax'].append(vals[2])
                values['pnames'].append('end.' + name)
                values['pinits'].append(vals[3])
                values['pmin'].append(vals[4])
                values['pmax'].append(vals[5])

        if variant == 1:
            values['pmin'] = None
            values['pmax'] = None
            if not hasScale:
                values['pscale'] = None
        else:
            values['pscale'] = None

        return values

# vim: tabstop=4 shiftwidth=4 expandtab
