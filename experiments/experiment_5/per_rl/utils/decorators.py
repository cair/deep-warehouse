import sys
def KeepLocals(attribute, include):
    # https://stackoverflow.com/questions/38498159/accessing-original-function-variables-in-decorators

    def Wrapped(f, **kwargs):

        def locals_to_globals(self, *args, **kwargs):

            """
            Calls the function *func* with the specified arguments and keyword
            arguments and snatches its local frame before it actually executes.
            """
            frame = None
            trace = sys.gettrace()

            def snatch_locals(_frame, name, arg):
                nonlocal frame
                if frame is None and name == 'call':
                    frame = _frame
                    sys.settrace(trace)
                return trace
            sys.settrace(snatch_locals)
            try:
                result = f(self, *args, **kwargs)
            finally:
                sys.settrace(trace)

            # If there is anything to include to global scope
            #if include:  # Assume that include is set here
            saveloc = getattr(self, attribute)
            f_locals = frame.f_locals
            saveloc.update({k: f_locals[k] for k in include})
            return result

        return locals_to_globals
    return Wrapped
