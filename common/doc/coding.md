# Kirakira Coding

### Components

#### Render passes

To be added...

#### Tone mappers

Currently supported operators: ACES, Reinhard.

#### Camera and controlling

To be added...

#### Shading

Shading normal in world space always points to the outside of an object. We determine if an incident ray comes from outside of the object if its direction wi.z>0 in local space. 

#### BSDFs

To be added...

#### Tagged Pointer

To be added...

### Debug

#### Macros

For debug build, there will be a `KRR_DEBUG_BUILD` macro defined for all code files. See cmake file for detail.

