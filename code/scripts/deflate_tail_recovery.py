"""Optional tail-only recovery from a verified literal DEFLATE block anchor.

RFC1951 stored blocks end on byte boundaries; only the preceding32KiB is
needed for subsequent distance codes. Candidate output is never committed
unless the entire original-prefix + recovered-tail CRC and SHA both match.
"""
from pathlib import Path
import hashlib,struct,zlib

def data_offset(source, info):
    source.seek(info.header_offset);header=source.read(30)
    if header[:4]!=b'PK\x03\x04':raise ValueError('Invalid ZIP local header')
    name,extra=struct.unpack_from('<HH',header,26)
    return info.header_offset+30+name+extra

def find_anchors(partial, band, band_start, prefix_size):
    answers={}
    with Path(partial).open('rb') as local:
        for back in range(256,min(65536,prefix_size-256),4096):
            raw_anchor=prefix_size-back
            local.seek(raw_anchor);sample=local.read(256)
            if len(set(sample))<24:continue
            hit=band.find(sample)
            if hit<0 or band.find(sample,hit+1)>=0:continue
            for p in range(max(0,hit-65539),hit-3):
                length,inverse=struct.unpack_from('<HH',band,p)
                if not length or length^inverse!=65535:continue
                start,end=p+4,p+4+length
                if not start<=hit or hit+256>end or end>len(band):continue
                raw_start=raw_anchor-(hit-start);raw_end=raw_start+length
                if raw_start<0:continue
                known=min(raw_end,prefix_size)-raw_start
                if known<1024:continue
                local.seek(raw_start)
                if local.read(known)!=band[start:start+known]:continue
                key=(band_start+end,raw_end)
                answers[key]=dict(compressed_boundary=key[0],raw_boundary=raw_end,known_literal_bytes=known,
                    literal_suffix=band[start+max(0,prefix_size-raw_start):end] if raw_end>prefix_size else b'')
    return sorted(answers.values(),key=lambda a:-a['known_literal_bytes'])

def validated_tail(source,info,partial,expected_sha,progress=lambda x:None):
    """Return a separately stored, fully validated tail; leave prefix unchanged."""
    partial=Path(partial);prefix_size=partial.stat().st_size;offset=data_offset(source,info)
    saving=max(0,info.file_size-info.compress_size)
    begin=max(0,prefix_size-saving-8*1024**2);end=min(info.compress_size,prefix_size+2*1024**2)
    source.seek(offset+begin);band=source.read(end-begin)
    anchors=find_anchors(partial,band,begin,prefix_size)
    if not anchors:raise ValueError('No trustworthy literal block anchor; saved prefix unchanged')
    prefix_hash=hashlib.sha256();prefix_crc=0
    with partial.open('rb') as f:
        while block:=f.read(8*1024**2):prefix_hash.update(block);prefix_crc=zlib.crc32(block,prefix_crc)
    errors=[]
    for index,anchor in enumerate(anchors[:4]):
        output=partial.with_name(partial.name+f'.tail_{anchor["compressed_boundary"]}')
        if output.exists():raise FileExistsError('Candidate recovery file already exists; inspect rather than overwrite')
        digest=prefix_hash.copy();crc=prefix_crc;generated=anchor['raw_boundary'];tail_bytes=0
        with partial.open('rb') as local:
            before=min(generated,prefix_size);local.seek(max(0,before-32768));history=local.read(min(32768,before))
        literal=anchor['literal_suffix']
        history=(history+literal)[-32768:]
        decoder=zlib.decompressobj(-15,zdict=history)
        try:
            with output.open('xb') as tail,partial.open('rb') as local:
                if literal:tail.write(literal);digest.update(literal);crc=zlib.crc32(literal,crc);tail_bytes+=len(literal)
                source.seek(offset+anchor['compressed_boundary']);remaining=info.compress_size-anchor['compressed_boundary']
                while remaining:
                    block=source.read(min(32*1024**2,remaining));remaining-=len(block)
                    if not block:raise ValueError('Truncated compressed tail')
                    pending=block
                    while pending:
                        expanded=decoder.decompress(pending,64*1024**2);pending=decoder.unconsumed_tail
                        compare=min(len(expanded),max(0,prefix_size-generated))
                        if compare:
                            local.seek(generated)
                            if local.read(compare)!=expanded[:compare]:raise ValueError('Restarted stream disagrees with existing prefix')
                        new=expanded[compare:]
                        if new:tail.write(new);digest.update(new);crc=zlib.crc32(new,crc);tail_bytes+=len(new)
                        generated+=len(expanded)
                        if generated>info.file_size:raise ValueError('Restarted stream exceeds expected size')
                    progress(dict(phase='recovering_missing_tail',raw_verified_bytes=prefix_size+tail_bytes,member_bytes=info.file_size,
                                  preserved_partial_bytes=prefix_size,appended_bytes=0,validated_candidate_tail_bytes=tail_bytes))
                tail.flush()
            if not decoder.eof or decoder.unused_data or generated!=info.file_size or prefix_size+tail_bytes!=info.file_size:
                raise ValueError('Restarted DEFLATE stream boundary/length mismatch')
            if (crc&0xffffffff)!=info.CRC or digest.hexdigest()!=expected_sha:
                raise ValueError('Full prefix+tail CRC/SHA mismatch; original prefix retained')
            return dict(tail_path=str(output.resolve()),tail_bytes=tail_bytes,preserved_partial_bytes=prefix_size,
                 original_partial_sha256=prefix_hash.hexdigest(),sha256=digest.hexdigest(),crc32=crc&0xffffffff,
                 compressed_restart=anchor['compressed_boundary'],raw_restart=anchor['raw_boundary'],candidate_attempt=index+1)
        except (ValueError,zlib.error) as error:
            errors.append(str(error))
    raise ValueError('All candidate anchors rejected with original prefix unchanged: '+repr(errors))
