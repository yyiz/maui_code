NVCC = nvcc
CUDAFLAGS = -arch=sm_60

SRCDIR = src
INCDIR = inc
OBJDIR = obj

SRCEXT = cpp
OBJEXT = o

SOURCES     := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS     := $(patsubst $(SRCDIR)/%,$(OBJDIR)/%,$(SOURCES:.$(SRCEXT)=.$(OBJEXT)))

maui: $(OBJECTS)
	$(NVCC) $(CUDAFLAGS) $(OBJECTS) -o maui

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(NVCC) $(CUDAFLAGS) -x cu -I$(INCDIR) -dc $< -o $@

clean:
	rm $(OBJDIR)/*.o maui
