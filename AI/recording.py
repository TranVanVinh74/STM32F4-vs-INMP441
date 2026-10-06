import serial
import struct
import csv
import time
import os


# ============================================================
# CONFIG
# ============================================================

PORT = "COM4"
BAUDRATE = 921600

NUM_BANDS = 16
MAX_SOURCES = 3

# ------------------------------------------------------------
# Hien tai chi thu SPEECH
# LABEL = 1 => KEEP / VALID SOUND
# ------------------------------------------------------------

LABEL = 0
SOUND_TYPE = "fan"

# ------------------------------------------------------------
# Moi lan thu tang SESSION_ID len
#
# 1 -> speech_01.csv
# 2 -> speech_02.csv
# 3 -> speech_03.csv
# ------------------------------------------------------------

SESSION_ID = 9

# ------------------------------------------------------------
# Minimum valid FFT bins
# ------------------------------------------------------------

MIN_BIN_COUNT = 3


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

CLASS_FOLDER = "noise"

OUTPUT_DIR = os.path.join(
    "final_test_raw",
    CLASS_FOLDER,
    SOUND_TYPE
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CREATE FILE NAME
# ============================================================

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    f"{SOUND_TYPE}_{SESSION_ID:02d}.csv"
)


# ============================================================
# CHECK FILE EXISTS
# ============================================================

if os.path.exists(OUTPUT_FILE):

    print()
    print("============================================")
    print("ERROR")
    print("============================================")

    print(
        "File already exists:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        "Please change SESSION_ID."
    )

    raise SystemExit


# ============================================================
# SERIAL INIT
# ============================================================

ser = serial.Serial(
    port=PORT,
    baudrate=BAUDRATE,
    timeout=1
)

time.sleep(1)

ser.reset_input_buffer()


print()
print("============================================")
print("STM32 SPEECH DATASET COLLECTION")
print("============================================")

print(
    "Serial port :",
    PORT
)

print(
    "Baudrate    :",
    BAUDRATE
)

print()

print(
    "Class       : KEEP / VALID SOUND"
)

print(
    "Sound type  : speech"
)

print(
    "Session ID  :",
    SESSION_ID
)

print()

print(
    "Output file:"
)

print(
    OUTPUT_FILE
)

print()

print(
    "Speech source can be placed at arbitrary angles."
)

print(
    "Press Ctrl+C to stop."
)

print()


# ============================================================
# CSV HEADER
# ============================================================

header = [
    "timestamp",
    "sound_type",
    "detected_angle",
    "bin_count"
]

for i in range(NUM_BANDS):

    header.append(
        f"band_{i}"
    )

header.append(
    "label"
)


# ============================================================
# CREATE CSV FILE
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(
        header
    )


# ============================================================
# READ EXACT N BYTES
# ============================================================

def read_exact(n):

    data = bytearray()

    while len(data) < n:

        chunk = ser.read(
            n - len(data)
        )

        if len(chunk) == 0:

            return None

        data.extend(
            chunk
        )

    return bytes(data)


# ============================================================
# READ ONE STM32 PACKET
# ============================================================

def read_packet():

    # --------------------------------------------------------
    # Search start byte 0xBB
    # --------------------------------------------------------

    while True:

        byte = ser.read(1)

        if not byte:
            return None

        if byte[0] == 0xBB:
            break


    # --------------------------------------------------------
    # Active source count
    # --------------------------------------------------------

    data = read_exact(1)

    if data is None:
        return None


    active_count = data[0]


    if active_count > MAX_SOURCES:

        print(
            "Invalid source count:",
            active_count
        )

        return None


    sources = []


    # --------------------------------------------------------
    # Read each source
    # --------------------------------------------------------

    for source_index in range(
        active_count
    ):

        # ====================================================
        # ANGLE
        # ====================================================

        angle_data = read_exact(2)

        if angle_data is None:
            return None


        angle = struct.unpack(
            "<H",
            angle_data
        )[0]


        # ====================================================
        # BIN COUNT
        # ====================================================

        bin_data = read_exact(1)

        if bin_data is None:
            return None


        bin_count = bin_data[0]


        # ====================================================
        # 16 FEATURES
        # ====================================================

        feature_data = read_exact(
            NUM_BANDS * 4
        )

        if feature_data is None:
            return None


        features = struct.unpack(
            "<" + "f" * NUM_BANDS,
            feature_data
        )


        sources.append(
            {
                "angle": angle,
                "bin_count": bin_count,
                "features": features
            }
        )


    # --------------------------------------------------------
    # END BYTE 0x55
    # --------------------------------------------------------

    end_byte = read_exact(1)

    if end_byte is None:
        return None


    if end_byte[0] != 0x55:

        print(
            "Packet error: expected 0x55, got",
            hex(end_byte[0])
        )

        return None


    return sources


# ============================================================
# SAVE ONE SOURCE
# ============================================================

def save_source(source):

    row = [
        time.time(),
        SOUND_TYPE,
        source["angle"],
        source["bin_count"]
    ]

    row.extend(
        source["features"]
    )

    row.append(
        LABEL
    )


    with open(
        OUTPUT_FILE,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            row
        )


# ============================================================
# MAIN LOOP
# ============================================================

try:

    sample_count = 0
    skipped_count = 0
    packet_count = 0


    while True:

        sources = read_packet()


        if sources is None:
            continue


        packet_count += 1


        # active_count = 0
        if len(sources) == 0:
            continue


        for i, source in enumerate(
            sources
        ):

            angle = source[
                "angle"
            ]

            bin_count = source[
                "bin_count"
            ]

            features = source[
                "features"
            ]


            # =================================================
            # PRINT
            # =================================================

            print(
                f"Source {i + 1} | "
                f"Angle={angle:3d} deg | "
                f"Bins={bin_count:2d}"
            )


            print(
                "Features:",
                "["
                +
                ", ".join(
                    f"{x:.2e}"
                    for x in features
                )
                +
                "]"
            )


            # =================================================
            # FILTER
            # =================================================

            if bin_count >= MIN_BIN_COUNT:

                save_source(
                    source
                )

                sample_count += 1

                print(
                    f"--> Saved #{sample_count}"
                )

            else:

                skipped_count += 1

                print(
                    "--> Skip: too few valid FFT bins"
                )


            print()


# ============================================================
# CTRL+C
# ============================================================

except KeyboardInterrupt:

    print()
    print("============================================")
    print("COLLECTION STOPPED")
    print("============================================")

    print(
        "Saved samples :",
        sample_count
    )

    print(
        "Skipped       :",
        skipped_count
    )

    print(
        "Packets       :",
        packet_count
    )

    print()

    print(
        "Dataset file:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# CLOSE SERIAL
# ============================================================

finally:

    ser.close()

    print()
    print(
        "Serial closed."
    )