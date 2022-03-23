# Simple script to disable ASLR and make .nv_fatb sections read-only
# Requires: pefile  ( python -m pip install pefile )
# Usage:  fixNvPe.py --input path/to/*.dll

import argparse
import pefile
import glob
import os
import shutil
import sys
import logging


def main(args):
    logger = logging.getLogger(__name__)
    failures = []
    for file in glob.glob(args.input, recursive=args.recursive):
        logging.info(f"\n---\nChecking {file}...")
        pe = pefile.PE(file, fast_load=True)
        nvbSect = [
            section for section in pe.sections
            if section.Name.decode().startswith(".nv_fatb")
        ]
        if len(nvbSect) == 1:
            sect = nvbSect[0]
            size = sect.Misc_VirtualSize
            aslr = pe.OPTIONAL_HEADER.IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE
            writable = 0 != (
                sect.Characteristics &
                pefile.SECTION_CHARACTERISTICS['IMAGE_SCN_MEM_WRITE'])
            logger.info(
                f"Found NV FatBin! Size: {size/1024/1024:0.2f}MB  ASLR: {aslr}  Writable: {writable}"
            )
            if (writable or aslr) and size > 0:
                logger.info("- Modifying DLL")
                if args.backup:
                    bakFile = f"{file}_bak"
                    logger.info(f"- Backing up [{file}] -> [{bakFile}]")
                    if os.path.exists(bakFile):
                        logger.warn(
                            f"- Backup file already exists ({bakFile}), not modifying file! Delete the 'bak' to allow modification"
                        )
                        failures.append(file)
                        continue
                    try:
                        shutil.copy2(file, bakFile)
                    except Exception as e:
                        logger.error(
                            f"- Failed to create backup! [{str(e)}], not modifying file!"
                        )
                        failures.append(file)
                        continue
                # Disable ASLR for DLL, and disable writing for section
                pe.OPTIONAL_HEADER.DllCharacteristics &= ~pefile.DLL_CHARACTERISTICS[
                    'IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE']
                sect.Characteristics = sect.Characteristics & ~pefile.SECTION_CHARACTERISTICS[
                    'IMAGE_SCN_MEM_WRITE']
                try:
                    newFile = f"{file}_mod"
                    logger.info(f"- Writing modified DLL to [{newFile}]")
                    pe.write(newFile)
                    pe.close()
                    logger.info(f"- Moving modified DLL to [{file}]")
                    os.remove(file)
                    shutil.move(newFile, file)
                except Exception as e:
                    logger.error(f"- Failed to write modified DLL! [{str(e)}]")
                    failures.append(file)
                    continue

    logger.info("\n\nDone!")
    if len(failures) > 0:
        logger.error("These files needed modification but failed: ")
        for failure in failures:
            logger.error(f" - {failure}")


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Disable ASLR and make .nv_fatb sections read-only",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', help="Glob to parse", default=None)
    parser.add_argument('--backup',
                        help="Backup modified files",
                        default=False,
                        required=False)
    parser.add_argument('--recursive',
                        '-r',
                        default=False,
                        action='store_true',
                        help="Recurse into subdirectories")

    return parser.parse_args()


###############################
# program entry point
#
if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    clargs = parseArgs()

    if clargs.input is None:
        clargs.input = os.path.join(os.path.dirname(sys.executable),
                                    r"Lib\site-packages\torch\lib\*.dll")
    logger.info(f"Checking for DLLs in {clargs.input}")
    assert os.path.exists(os.path.dirname(clargs.input))
    main(clargs)
