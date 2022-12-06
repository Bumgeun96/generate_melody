import pretty_midi

example_file = 'example.midi'
def notes_to_midi(data,out_file=example_file,instrument_name='Acoustic Grand Piano'):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))

    prev_start = 0
    for i, note in enumerate(data):
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(
            velocity=100,
            pitch=int(note['pitch']),
            start=start,
            end=end,
            )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm
