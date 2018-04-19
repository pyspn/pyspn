#!/usr/bin/env python3

import bitarray
import logging


class Valid(object):
    def __init__(self):
        self.layer_type = None
        self.name = None
        self.num = None

        def check_valid(self, layer):
            sum = bitarray()
            parent = bitarray()
            for node in layer._nodes:  # TODO: Make sure that there is
                #  function which returns nodelist
                sum &= node._bitarray()
                parent |= node._bitarray()
            if layer._type == "sum":
                if sum.count() > 0:  # They belong to the same scope
                    logging.warning("Layer", layer._number, "is valid")
                    return parent
                else:
                    logging.warning("Layer", layer._number, "is invalid")
                    raise Exception("The current layer is invalid")
            elif layer._type == "product":
                if sum.count() == 0:
                    logging.error("Layer", layer._number, "is valid")
                    return parent
                else:
                    logging.warning("Layer", layer._number, "is invalid")
                    raise Exception("The current layer is invalid")
            else:
                logging.error("The layer type mentioned is invalid")
                raise Exception("")
