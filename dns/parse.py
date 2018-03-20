import argparse
import scapy.layers.dns
import scapy.sendrecv


def dnsonly(pkt):
    print(pkt.summary())


def main():
    parser = argparse.ArgumentParser(description="Parse DNS packets")
    parser.add_argument("pcap", metavar="PCAP", type=str,
                        help="Traffic dump.")

    args = parser.parse_args()

    scapy.sendrecv.sniff(
        offline=args.pcap,
        store=False,
        lfilter=lambda p: p.haslayer(scapy.layers.dns.DNS),
        prn=dnsonly)


if __name__ == "__main__":
    main()
