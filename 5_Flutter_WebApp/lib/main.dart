import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'dart:async';
import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter/gestures.dart' show DragStartBehavior;
import 'package:flutter/widgets.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SmartDota',
      theme: ThemeData(
        // This is the theme of your application.
        //
        // Try running your application with "flutter run". You'll see the
        // application has a blue toolbar. Then, without quitting the app, try
        // changing the primarySwatch below to Colors.green and then invoke
        // "hot reload" (press "r" in the console where you ran "flutter run",
        // or simply save your changes to "hot reload" in a Flutter IDE).
        // Notice that the counter didn't reset back to zero; the application
        // is not restarted.
        primarySwatch: Colors.grey,
        // This makes the visual density adapt to the platform that you run
        // the app on. For desktop platforms, the controls will be smaller and
        // closer together (more dense) than on mobile platforms.
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: MyHomePage(),
    );
  }
}

class SingleChildScrollViewWithScrollbar extends StatefulWidget {
  const SingleChildScrollViewWithScrollbar({
    Key key,
    this.scrollDirection = Axis.vertical,
    this.reverse = false,
    this.padding,
    this.primary,
    this.physics,
    this.controller,
    this.child,
    this.dragStartBehavior = DragStartBehavior.down,
    this.scrollbarColor,
    this.scrollbarThickness = 6.0,
  }) : super(key: key);

  final Axis scrollDirection;
  final bool reverse;
  final EdgeInsets padding;
  final bool primary;
  final ScrollPhysics physics;
  final ScrollController controller;
  final Widget child;
  final DragStartBehavior dragStartBehavior;
  final Color scrollbarColor;
  final double scrollbarThickness;

  @override
  _SingleChildScrollViewWithScrollbarState createState() =>
      _SingleChildScrollViewWithScrollbarState();
}

class _SingleChildScrollViewWithScrollbarState
    extends State<SingleChildScrollViewWithScrollbar> {
  AlwaysVisibleScrollbarPainter _scrollbarPainter;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    rebuildPainter();
  }

  @override
  void didUpdateWidget(SingleChildScrollViewWithScrollbar oldWidget) {
    super.didUpdateWidget(oldWidget);
    rebuildPainter();
  }

  void rebuildPainter() {
    final theme = Theme.of(context);
    _scrollbarPainter = AlwaysVisibleScrollbarPainter(
      color: widget.scrollbarColor ?? theme.highlightColor.withOpacity(1.0),
      textDirection: Directionality.of(context),
      thickness: widget.scrollbarThickness,
    );
  }

  @override
  void dispose() {
    _scrollbarPainter?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return RepaintBoundary(
      child: CustomPaint(
        foregroundPainter: _scrollbarPainter,
        child: RepaintBoundary(
          child: SingleChildScrollView(
            scrollDirection: widget.scrollDirection,
            reverse: widget.reverse,
            padding: widget.padding,
            primary: widget.primary,
            physics: widget.physics,
            controller: widget.controller,
            dragStartBehavior: widget.dragStartBehavior,
            child: Builder(
              builder: (BuildContext context) {
                _scrollbarPainter.scrollable = Scrollable.of(context);
                return widget.child;
              },
            ),
          ),
        ),
      ),
    );
  }
}

class AlwaysVisibleScrollbarPainter extends ScrollbarPainter {
  AlwaysVisibleScrollbarPainter({
    @required Color color,
    @required TextDirection textDirection,
    @required double thickness,
  }) : super(
          color: color,
          textDirection: textDirection,
          thickness: thickness,
          fadeoutOpacityAnimation: const AlwaysStoppedAnimation(1.0),
        );

  ScrollableState _scrollable;

  ScrollableState get scrollable => _scrollable;

  set scrollable(ScrollableState value) {
    _scrollable?.position?.removeListener(_onScrollChanged);
    _scrollable = value;
    _scrollable?.position?.addListener(_onScrollChanged);
    _onScrollChanged();
  }

  void _onScrollChanged() {
    update(_scrollable.position, _scrollable.axisDirection);
  }

  @override
  void dispose() {
    _scrollable?.position?.removeListener(notifyListeners);
    super.dispose();
  }
}

class MyHomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    // This method is rerun every time setState is called, for instance as done
    // by the _incrementCounter method above.
    //
    // The Flutter framework has been optimized to make rerunning build methods
    // fast, so that you can just rebuild anything that needs updating rather
    // than having to individually change instances of widgets.
    return Scaffold(
      appBar: AppBar(
        // Here we take the value from the MyHomePage object that was created by
        // the App.build method, and use it to set our appbar title.
        //title: Text('Home',style: TextStyle(color: Colors.white)),
        title: Row(children: [
          Align(
            alignment: Alignment.centerLeft,
            child: Text(
              'Home',
              style: TextStyle(color: Colors.white),
            ),
          ),
          Padding(
            padding: EdgeInsets.fromLTRB(20, 0, 0, 0),
            child: Align(
              alignment: Alignment.centerLeft,
              child: Text(
                'How Does This work?',
                style: TextStyle(color: Colors.white),
              ),
            ),
          )
        ]),
        backgroundColor: Color(0xFF151026),
      ),
      body: Align(
          alignment: Alignment.topCenter,
          // Center is a layout widget. It takes a single child and positions it
          // in the middle of the parent.
          child: Container(
              child: Scrollbar(
                  child: SingleChildScrollViewWithScrollbar(
            child: HomePageCol(),
          )))),
      // This trailing comma makes auto-formatting nicer for build methods.
    );
  }
}

List<bool> _side_selections = [true, false];
List<bool> _role_selections = [true, false, false, false, false];
List<int> _radiant_picks = [0, 0, 0, 0, 0];
List<int> _dire_picks = [0, 0, 0, 0, 0];
List<int> _pred_hero_id = [-1];
List<double> _pred_hero_chance = [-1];
List<Hero_image_picked> _radiant_picks_images = List.filled(
    5,
    Hero_image_picked(
      image: '',
      name: '',
      ID: 0,
    ));
List<Hero_image_picked> _dire_picks_images = List.filled(
    5,
    Hero_image_picked(
      image: '',
      name: '',
      ID: 0,
    ));

class HomePageCol extends StatefulWidget {
  @override
  _HomePageCol createState() => _HomePageCol();
}

class _HomePageCol extends State<HomePageCol> {
  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Padding(padding: EdgeInsets.fromLTRB(0, 15, 0, 0), child: Text('Side')),
        Padding(
          padding: EdgeInsets.fromLTRB(0, 10, 0, 0),
          child: ToggleButtons(
            children: [
              Padding(
                padding: EdgeInsets.fromLTRB(30, 0, 30, 0),
                child:
                    Align(alignment: Alignment.center, child: Text('Radiant')),
              ),
              Padding(
                padding: EdgeInsets.fromLTRB(30, 0, 30, 0),
                child: Align(alignment: Alignment.center, child: Text('Dire')),
              ),
            ],
            onPressed: (int index) {
              setState(() {
                _side_selections[0] = false;
                _side_selections[1] = false;
                _side_selections[index] = !_side_selections[index];
              });
            },
            isSelected: _side_selections,
          ),
        ),
        Padding(padding: EdgeInsets.fromLTRB(0, 15, 0, 0), child: Text('Role')),
        Padding(
          padding: EdgeInsets.fromLTRB(0, 10, 0, 0),
          child: ToggleButtons(
            children: [
              Padding(
                padding: EdgeInsets.fromLTRB(30, 0, 30, 0),
                child: Align(
                    alignment: Alignment.center, child: Text('Hard Support')),
              ),
              Padding(
                padding: EdgeInsets.fromLTRB(30, 0, 30, 0),
                child:
                    Align(alignment: Alignment.center, child: Text('Support')),
              ),
              Padding(
                padding: EdgeInsets.fromLTRB(30, 0, 30, 0),
                child:
                    Align(alignment: Alignment.center, child: Text('Offlane')),
              ),
              Padding(
                padding: EdgeInsets.fromLTRB(30, 0, 30, 0),
                child: Align(alignment: Alignment.center, child: Text('Carry')),
              ),
              Padding(
                padding: EdgeInsets.fromLTRB(30, 0, 30, 0),
                child: Align(alignment: Alignment.center, child: Text('Mid')),
              ),
            ],
            onPressed: (int index) {
              setState(() {
                _role_selections[0] = false;
                _role_selections[1] = false;
                _role_selections[2] = false;
                _role_selections[3] = false;
                _role_selections[4] = false;
                _role_selections[index] = true;
              });
            },
            isSelected: _role_selections,
          ),
        ),
        Padding(
            padding: EdgeInsets.fromLTRB(0, 15, 0, 15), child: Text('Heroes')),
        Hero_list_loader(),
      ],
    );
  }
}

Future<String> loadAsset() async {
  return await rootBundle.loadString('dota_heroes.json');
}

class Hero_list_loader extends StatefulWidget {
  @override
  _Hero_list_loader createState() => _Hero_list_loader();
}

class _Hero_list_loader extends State<Hero_list_loader> {
  @override
  Widget build(BuildContext context) {
    return FutureBuilder(
      future: loadAsset(),
      builder: (context, snapshot) {
        if (snapshot.hasData) {
          //Map user = jsonDecode(snapshot.data)
          var hero_data = json.decode(snapshot.data);
          hero_data = hero_data['heroes'];
          List<Hero_image> myList = List<Hero_image>();
          for (var m in hero_data) {
            myList.add(Hero_image(
                image: "hero_images/" + m['pic'],
                //image: m['pic'],
                name: m['name'],
                ID: m['id']));
          }
          myList.sort((a, b) => a.name.compareTo(b.name));
          return Hero_searcher(heroes: myList);
        } else
          return Text('Loading');
      },
    );
  }
}

class Hero_searcher extends StatefulWidget {
  final List<Hero_image> heroes;
  const Hero_searcher({Key key, this.heroes}) : super(key: key);
  @override
  _Hero_searcher createState() => _Hero_searcher();
}

class _Hero_searcher extends State<Hero_searcher> {
  String Textdata = "";
  List<Hero_image> heroes_to_show = [Hero_image(image: '', name: "", ID: -1)];

  void _update() {
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    if (Textdata == "") {
      heroes_to_show = List<Hero_image>();
      for (var hero in widget.heroes) {
        var image = hero.image;
        var id = hero.ID;
        var name = hero.name;
        heroes_to_show.add(Hero_image(
          image: image,
          name: name,
          ID: id,
          updateparent: _update,
        ));
      }
      return Column(children: [
        Padding(
          padding: EdgeInsets.fromLTRB(0, 0, 0, 10),
          child: Container(
            child: TextFormField(
              onChanged: (text) {
                Textdata = text;
                setState(() {});
              },
              decoration: new InputDecoration(hintText: "Search Heroes"),
            ),
            width: 300,
          ),
        ),
        Wrap(
          children: heroes_to_show,
        ),
        Heroes_picked(),
        Padding(
            padding: EdgeInsets.fromLTRB(0, 25, 0, 10),
            child: SizedBox(
                width: 100,
                height: 50,
                child: FlatButton(
                  color: Colors.yellow,
                  child: Text('Predict'),
                  onPressed: () {
                    predict_with_data(_update);
                    setState(() {});
                  },
                ))),
        prediction_data(heroes: widget.heroes),
      ]);
    } else
      heroes_to_show = List<Hero_image>();
    for (var hero in widget.heroes) {
      if (hero.name.toLowerCase().contains(Textdata.toLowerCase())) {
        var image = hero.image;
        var id = hero.ID;
        var name = hero.name;
        heroes_to_show.add(Hero_image(
          image: image,
          name: name,
          ID: id,
          updateparent: _update,
        ));
      }
    }
    _update();
    if (heroes_to_show.length == 0)
      heroes_to_show = [
        Hero_image(image: '', name: 'No hero with this name', ID: -1)
      ];
    return Column(children: [
      Padding(
        padding: EdgeInsets.fromLTRB(0, 0, 0, 10),
        child: Container(
          child: TextFormField(
            onChanged: (text) {
              Textdata = text;
              setState(() {});
            },
            decoration: new InputDecoration(hintText: "Search Heroes"),
          ),
          width: 300,
        ),
      ),
      Wrap(
        children: heroes_to_show,
      ),
      Heroes_picked(),
      Padding(
          padding: EdgeInsets.fromLTRB(0, 25, 0, 10),
          child: SizedBox(
              width: 100,
              height: 50,
              child: FlatButton(
                color: Colors.yellow,
                child: Text('Predict'),
                onPressed: () {
                  predict_with_data(_update);
                  setState(() {});
                },
              ))),
      prediction_data(heroes: widget.heroes),
    ]);
  }
}

Future<String> predict_with_data(Function update) async {
  var request = "http://xxx.herokuapp.com/api/v1/predictor?";
  for (var i = 0; i < 5; i++) {
    request = request +
        'r' +
        (i + 1).toString() +
        '=' +
        _radiant_picks[i].toString() +
        '&';
    request = request +
        'd' +
        (i + 1).toString() +
        '=' +
        _dire_picks[i].toString() +
        '&';
  }
  var side = "";
  if (_side_selections[0] == true)
    side = "radiant";
  else
    side = "dire";
  request = request + 'side=' + side + '&';
  var role = "";
  if (_role_selections[0] == true)
    role = 'hard_support';
  else if (_role_selections[1] == true)
    role = 'support';
  else if (_role_selections[2] == true)
    role = 'offlane';
  else if (_role_selections[3] == true)
    role = 'mid_lane';
  else
    role = 'carry';
  request = request + 'role=' + role;
  var res = await http.get(request);
  var data = json.decode(res.body);
  _pred_hero_id = List<int>();
  _pred_hero_chance = List<double>();
  for (var item in data) {
    _pred_hero_id.add(item[0]);
    _pred_hero_chance.add(item[1]);
  }
  update();
  return 'Success';
}

class prediction_data extends StatefulWidget {
  final List<Hero_image> heroes;
  const prediction_data({Key key, this.heroes}) : super(key: key);
  @override
  _prediction_data createState() => _prediction_data();
}

class _prediction_data extends State<prediction_data> {
  @override
  Widget build(BuildContext context) {
    if (_pred_hero_id.contains(-1))
      return Text('');
    else {
      List<SizedBox> cards = List<SizedBox>();
      for (var i = 0; i < _pred_hero_id.length; i++) {
        var img_url = '';
        var name = '';
        for (var hero in widget.heroes)
          if (hero.ID == _pred_hero_id[i]) {
            img_url = hero.image;
            name = hero.name;
          }
        var one_card = SizedBox(
          width: 500,
          child:Card(
            child: Column(mainAxisSize: MainAxisSize.min, children: <Widget>[
          ListTile(
            leading: Image.asset(img_url),
            title: Text(name),
            subtitle: Text(_pred_hero_chance[i].toString()),
          )
        ])));
        cards.add(one_card);
      }
      return Column(children: cards,mainAxisSize: MainAxisSize.min,);
    }
  }
}

class Heroes_picked extends StatefulWidget {
  @override
  _Heroes_picked createState() => _Heroes_picked();
}

class _Heroes_picked extends State<Heroes_picked> {
  @override
  Widget build(BuildContext context) {
    return Wrap(
      children: [
        Padding(
            padding: EdgeInsets.fromLTRB(10, 10, 10, 0),
            child: Column(mainAxisSize: MainAxisSize.min, children: [
              Text('Radiant picks'),
              Row(
                  mainAxisSize: MainAxisSize.min,
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: _radiant_picks_images)
            ])),
        Padding(
          padding: EdgeInsets.fromLTRB(10, 10, 10, 0),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Text('Dire picks'),
            Row(
                mainAxisSize: MainAxisSize.min,
                mainAxisAlignment: MainAxisAlignment.center,
                children: _dire_picks_images)
          ]),
        )
      ],
    );
  }
}

class Hero_image_picked extends StatefulWidget {
  final String image;
  final String name;
  final int ID;
  final Function updateparent;
  const Hero_image_picked(
      {Key key, this.image, this.name, this.ID, this.updateparent})
      : super(key: key);
  @override
  _Hero_image_picked createState() => _Hero_image_picked();
}

class _Hero_image_picked extends State<Hero_image_picked> {
  @override
  Widget build(BuildContext context) {
    if (widget.ID == 0) {
      return Image.asset('/not_picked.jpg');
    } else {
      return Stack(children: [
        Image.asset(widget.image),
        Positioned(
          child: IconButton(
            icon: Icon(Icons.close),
            color: Colors.white,
            onPressed: () {
              var ID = widget.ID;
              if (_radiant_picks.contains(ID)) {
                var index = _radiant_picks.indexOf(ID);
                for (var i = index + 1; i <= 4; i++) {
                  _radiant_picks[i - 1] = _radiant_picks[i];
                  _radiant_picks_images[i - 1] = _radiant_picks_images[i];
                }
              }
              if (_dire_picks.contains(ID)) {
                var index = _dire_picks.indexOf(ID);
                for (var i = index + 1; i <= 4; i++) {
                  _dire_picks[i - 1] = _dire_picks[i];
                  _dire_picks_images[i - 1] = _dire_picks_images[i];
                }
              }
              widget.updateparent();
              setState(() {});
            },
          ),
          top: -5,
          left: 90,
        ),
      ]);
    }
  }
}

class Hero_image extends StatefulWidget {
  //final int id;
  final String image;
  final String name;
  final int ID;
  final Function updateparent;

  const Hero_image({Key key, this.image, this.name, this.ID, this.updateparent})
      : super(key: key);
  @override
  _Hero_image createState() => _Hero_image();
}

class _Hero_image extends State<Hero_image> {
  Function updateparent;
  bool hovering = false;

  void _mouseEnter(bool hover) {
    setState(() {
      hovering = hover;
    });
  }

  void _pick_radiant(int ID) {
    if (_dire_picks.contains(ID)) {
      var index = _dire_picks.indexOf(ID);
      for (var i = index + 1; i <= 4; i++) {
        _dire_picks[i - 1] = _dire_picks[i];
        _dire_picks_images[i - 1] = _dire_picks_images[i];
      }
    }
    if (!_radiant_picks.contains(ID)) {
      var counter = 0;
      for (var item in _radiant_picks) {
        if (counter == 4) {
          break;
        }
        if (item == 0) {
          _radiant_picks[counter] = ID;
          _radiant_picks_images[counter] = Hero_image_picked(
            image: widget.image,
            name: widget.name,
            ID: widget.ID,
            updateparent: widget.updateparent,
          );
          break;
        }
        counter++;
      }
    }
    widget.updateparent();
  }

  void _pick_dire(int ID) {
    if (_radiant_picks.contains(ID)) {
      var index = _radiant_picks.indexOf(ID);
      for (var i = index + 1; i <= 4; i++) {
        _radiant_picks[i - 1] = _radiant_picks[i];
        _radiant_picks_images[i - 1] = _radiant_picks_images[i];
      }
    }
    if (!_dire_picks.contains(ID)) {
      var counter = 0;
      for (var item in _dire_picks) {
        if (counter == 4) {
          break;
        }
        if (item == 0) {
          _dire_picks[counter] = ID;
          _dire_picks_images[counter] = Hero_image_picked(
            image: widget.image,
            name: widget.name,
            ID: widget.ID,
            updateparent: widget.updateparent,
          );
          break;
        }

        counter++;
      }
    }
    widget.updateparent();
    
  }

  @override
  Widget build(BuildContext context) {
    if (widget.ID == -1) {
      return Text(widget.name);
    }
    if (hovering == false)
      return MouseRegion(
        onEnter: (e) => _mouseEnter(true),
        onExit: (e) => _mouseEnter(false),
        child: Image.asset(widget.image),
      );
    else
      return MouseRegion(
          onEnter: (e) => _mouseEnter(true),
          onExit: (e) => _mouseEnter(false),
          child: Stack(
            children: [
              Opacity(child: Image.asset(widget.image), opacity: 0.5),
              Opacity(
                  child: SizedBox(
                    width: 69,
                    height: 72,
                    child: const DecoratedBox(
                      decoration: const BoxDecoration(color: Colors.green),
                    ),
                  ),
                  opacity: 0.5),
              Positioned(
                child: Opacity(
                    child: SizedBox(
                      width: 69,
                      height: 72,
                      child: const DecoratedBox(
                        decoration: const BoxDecoration(color: Colors.red),
                      ),
                    ),
                    opacity: 0.5),
                left: 69,
              ),
              Positioned(
                child: Opacity(
                    child: SizedBox(
                      width: 69,
                      height: 72,
                      child: FlatButton(
                        child: Text(
                          'Pick Radiant',
                          style: TextStyle(color: Colors.black),
                        ),
                        onPressed: () {
                          _pick_radiant(widget.ID);
                        },
                      ),
                    ),
                    opacity: 0.9),
                left: 0,
              ),
              Positioned(
                child: Opacity(
                    child: SizedBox(
                      width: 69,
                      height: 72,
                      child: FlatButton(
                        child: Text(
                          'Pick Dire',
                          style: TextStyle(color: Colors.black),
                        ),
                        onPressed: () {
                          _pick_dire(widget.ID);
                        },
                      ),
                    ),
                    opacity: 0.9),
                left: 69,
              ),
            ],
          ));
  }
}
