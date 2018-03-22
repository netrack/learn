import argparse
import csv
import scapy.layers.dns
import scapy.sendrecv
import math


def shannon(b):
    slen = len(b)

    freqs = (
        float(b.count(c)) / slen
        for c in set(b))

    return -sum((
        prob * math.log(prob, 2.0)
        for prob in freqs))


def writeto(writer, label):
    csvwriter = csv.writer(writer)
    csvwriter.writerow([
        "label",
        "qdcount",
        "ancount",
        "arcount",
        "nscount",
        "qd_qname_len",
        "qd_qname_shannon",
        "qd_qtype",
        "an_rrname_len",
        "an_rrname_shannon",
        "an_type",
        "an_ttl",
        "an_rdata_len",
        "an_rdata_shannon",
        "ar_rrname_len",
        "ar_rrname_shanonn",
        "ar_type",
        "ar_rdata_len",
        "ar_rdata_shannon",
    ])

    def _prn(pkt):
        row = dnsonly(pkt)
        csvwriter.writerow([label]+list(map(float, row)))

    return _prn

def dnsonly(pkt):
    dns = pkt.lastlayer()
    attrs = []

    # Number of questions.
    attrs.append(dns.qdcount)
    # Number of answers.
    attrs.append(dns.ancount)
    # Number of additional records.
    attrs.append(dns.arcount)
    # Numner of Name Server count.
    attrs.append(dns.nscount)

    # Length of the query name.
    attrs.append(len(dns.qd.qname))
    # Shannon entropy of the query name.
    attrs.append(shannon(dns.qd.qname))
    # Query type.
    attrs.append(dns.qd.qtype)

    if dns.ancount:
        attrs.append(len(dns.an.rrname))
        attrs.append(shannon(dns.an.rrname))
        attrs.append(dns.an.type)
        attrs.append(dns.an.ttl)

        if dns.an.rdata:
            attrs.append(len(dns.an.rdata))
            attrs.append(shannon(dns.an.rdata))
        else:
            attrs.append(0) # Answer RData length.
            attrs.append(0) # Answer RData shannon entropy.
    else:
        attrs.append(0) # Answer record RRName length.
        attrs.append(0) # Answer record RRName shannon entropy.
        attrs.append(0) # Answer record type.
        attrs.append(0) # Answer record TTL
        attrs.append(0) # Answer RData length.
        attrs.append(0) # Answer RData shannon entropy.

    if dns.arcount:
        attrs.append(len(dns.ar.rrname))
        attrs.append(shannon(dns.ar.rrname))
        attrs.append(dns.ar.type)

        if dns.ar.rdata:
            rdata = dns.ar.rdata
            if not isinstance(rdata, bytes):
                rdata = rdata[0].optdata
            attrs.append(len(rdata))
            attrs.append(shannon(rdata))
        else:
            attrs.append(0) # Additional record RData length.
            attrs.append(0) # Additional record RData shannon entropy.
    else:
        attrs.append(0) # Additional record RRName length.
        attrs.append(0) # Additional record RRName shanonn entropy.
        attrs.append(0) # Additional record RRName type.
        attrs.append(0) # Additional record RData length.
        attrs.append(0) # Additional record RData shannon entropy.

    return attrs


def main():
    parser = argparse.ArgumentParser(description="Parse DNS packets")
    parser.add_argument("pcap", metavar="PCAP", type=str,
                        help="Traffic dump.")
    parser.add_argument("out", metavar="OUT", type=str,
                        help="Output attributes.")
    parser.add_argument("label", metavar="label", type=int,
                        help="The class label.")

    args = parser.parse_args()

    # with open(args.out, "w", newline="") as csvfile:
    with open(args.out, "w") as csvfile:
        scapy.sendrecv.sniff(
            offline=args.pcap,
            store=False,
            lfilter=lambda p: p.haslayer(scapy.layers.dns.DNS),
            prn=writeto(csvfile, args.label))


if __name__ == "__main__":
    main()
