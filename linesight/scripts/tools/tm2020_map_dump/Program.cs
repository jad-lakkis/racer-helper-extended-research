// Dumps the geometry of a Trackmania 2020 .Map.Gbx whose road is made of embedded custom items.
//
//   dotnet run -- <map.Map.Gbx> <out dir>
//
// Writes to <out dir>:
//   map.json, placements.json   map name / size and where each item sits (position, yaw, pivot)
//   meshes.json                 one row per mesh part: item, material link, surface physic id, sizes
//   <part>.pos.f32, <part>.idx.i32   vertex positions (x, y, z little-endian floats) and triangle indices
//   embedded/                   the extracted embedded item files
// tm2020_centerline.py turns this into a centerline.
using System.Text.Json;
using GBX.NET;
using GBX.NET.Engines.Game;
using GBX.NET.Engines.GameData;
using GBX.NET.Engines.Plug;
using GBX.NET.LZO;

if (args.Length != 2)
{
    Console.Error.WriteLine("usage: tm2020_map_dump <map.Map.Gbx> <out dir>");
    return 1;
}

Gbx.LZO = new Lzo();
var map = (CGameCtnChallenge)Gbx.ParseNode(args[0])!;
var outDir = args[1];
var embedded = Path.Combine(outDir, "embedded");
Directory.CreateDirectory(embedded);
map.ExtractEmbeddedZipData(embedded);
var json = new JsonSerializerOptions { WriteIndented = true };

var placements = (map.AnchoredObjects ?? new List<CGameCtnAnchoredObject>()).Select(o => new
{
    item = o.ItemModel.Id,
    pos = new[] { o.AbsolutePositionInMap.X, o.AbsolutePositionInMap.Y, o.AbsolutePositionInMap.Z },
    yaw_pitch_roll = new[] { o.YawPitchRoll.X, o.YawPitchRoll.Y, o.YawPitchRoll.Z },
    pivot = new[] { o.PivotPosition.X, o.PivotPosition.Y, o.PivotPosition.Z },
    scale = o.Scale,
}).ToList();
File.WriteAllText(Path.Combine(outDir, "placements.json"), JsonSerializer.Serialize(placements, json));
File.WriteAllText(Path.Combine(outDir, "map.json"), JsonSerializer.Serialize(new
{
    name = map.MapName,
    size = new[] { map.Size.X, map.Size.Y, map.Size.Z },
    author = map.AuthorLogin,
    blocks = map.GetBlocks().Count(),
    baked_blocks = map.BakedBlocks?.Count() ?? 0,
}, json));

var meta = new List<object>();
foreach (var f in Directory.GetFiles(embedded, "*.Item.Gbx", SearchOption.AllDirectories))
{
    if (Gbx.ParseNode(f) is not CGameItemModel item) continue;
    if (item.EntityModel is not CGameCommonItemEntityModel ce || ce.StaticObject?.Mesh is not CPlugSolid2Model mesh || mesh.Visuals == null) continue;
    var rel = Path.GetRelativePath(embedded, f).Replace('\\', '/');
    var key = rel.Replace("/", "__").Replace(".Item.Gbx", "");
    var geoms = mesh.ShadedGeoms ?? Array.Empty<CPlugSolid2Model.ShadedGeom>();
    for (int vi = 0; vi < mesh.Visuals.Length; vi++)
    {
        if (mesh.Visuals[vi] is not CPlugVisualIndexedTriangles v || v.VertexStreams == null || v.VertexStreams.Count == 0) continue;
        var pos = v.VertexStreams[0].Positions;
        var idx = v.IndexBuffer?.Indices;
        if (pos == null || idx == null) continue;
        var sg = geoms.FirstOrDefault(g => g.VisualIndex == vi);
        string link = "", phys = "";
        if (sg != null && mesh.CustomMaterials != null && sg.MaterialIndex >= 0 && sg.MaterialIndex < mesh.CustomMaterials.Length)
        {
            var m = mesh.CustomMaterials[sg.MaterialIndex];
            link = m.MaterialUserInst?.Link ?? "";
            phys = m.MaterialUserInst?.SurfacePhysicId.ToString() ?? "";
        }
        var name = $"{key}__{vi}";
        using (var w = new BinaryWriter(File.Create(Path.Combine(outDir, name + ".pos.f32"))))
            foreach (var p in pos) { w.Write(p.X); w.Write(p.Y); w.Write(p.Z); }
        using (var w = new BinaryWriter(File.Create(Path.Combine(outDir, name + ".idx.i32"))))
            foreach (var i in idx) w.Write(i);
        meta.Add(new { item = rel, visual = vi, file = name, vertices = pos.Length, indices = idx.Length, link, surface_physic = phys });
    }
}
File.WriteAllText(Path.Combine(outDir, "meshes.json"), JsonSerializer.Serialize(meta, json));
Console.WriteLine($"{map.MapName}: {placements.Count} placed items, {meta.Count} mesh parts written to {outDir}");
return 0;
